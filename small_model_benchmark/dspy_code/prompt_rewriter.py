import dspy
from openai import OpenAI
from datasets import load_dataset
import json
from tqdm import tqdm
import argparse
from dspy.teleprompt import BootstrapFewShot
from dspy import Example
import os
from datetime import datetime
import time
import statistics
import random


def get_openai_client():
    """Get a fresh OpenAI client instance."""
    return OpenAI()  # This will use OPENAI_API_KEY from environment

def get_local_client():
    """Get a fresh local client instance."""
    return OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

class PoliteWordRemoverSignature(dspy.Signature):
    """Rewrites a prompt to remove polite words like 'please' and 'thank you' as long as they do not change the meaning of the prompt. If the polite words are not present or integral to the prompt, the prompt should remain unchanged."""
    initial_prompt: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step by step reasoning and calculations")
    rewritten_prompt: str = dspy.OutputField(desc="The rewritten prompt with polite words removed")

class PoliteWordRemoverPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PoliteWordRemoverSignature)
        # Fallback predictor with more explicit instructions
        self.fallback_predict = dspy.Predict(
            PoliteWordRemoverSignature,
            instructions="""You must provide both reasoning and rewritten prompt in your response.
First show your step-by-step reasoning, then on a new line write 'Rewritten: ' followed by the rewritten prompt.
Example format:
Reasoning: [your explanation]
Rewritten: [the rewritten prompt]"""
        )
    
    def forward(self, initial_prompt):
        """Process the input prompt and return the rewritten version."""
        try:
            return self.predict(initial_prompt=initial_prompt)
        except Exception as e:
            print("\nRetrying with more explicit instructions...")
            return self.fallback_predict(initial_prompt=initial_prompt)
    
    def __getstate__(self):
        """Customize pickling behavior."""
        state = self.__dict__.copy()
        # Remove unpicklable objects
        if 'predict' in state:
            del state['predict']
        return state
    
    def __setstate__(self, state):
        """Customize unpickling behavior."""
        self.__dict__.update(state)
        # Recreate unpicklable objects
        self.predict = dspy.Predict(PoliteWordRemoverSignature)

def custom_metric(pred, gold, trace=None):
    """Custom metric for evaluating prompt rewriting accuracy.
    
    For integral prompts (where polite words can't be removed):
    - Output should match input exactly
    
    For removable prompts (where polite words can be removed):
    - Output should be different from input
    - Output should maintain the core meaning
    - Output should not contain polite words
    """
    # Check if this is an integral prompt
    is_integral = gold["is_integral"]
    initial_prompt = gold["initial_prompt"]
    
    if is_integral:
        # For integral prompts, output should match input exactly
        return pred.rewritten_prompt.strip() == initial_prompt.strip()
    else:
        # For removable prompts:
        # 1. Output should be different from input
        if pred.rewritten_prompt.strip() == initial_prompt.strip():
            return False
            
        # 2. Output should not contain common polite words
        polite_words = ["please", "thank you", "thanks", "kindly", "would you mind", 
                       "could you", "would you", "if you don't mind", "appreciate"]
        for word in polite_words:
            if word.lower() in pred.rewritten_prompt.lower():
                return False
                
        # 3. Output should maintain core meaning
        # This is a simple check - in practice you might want a more sophisticated semantic similarity check
        return True

def generate_synthetic_data(num_examples=100):
    """Generate synthetic training data and save it to file."""
    # Create training data directory
    training_dir = os.path.join("training_data", "prompt_rewriter")
    os.makedirs(training_dir, exist_ok=True)
    
    # Check if directory is empty
    existing_files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
    if existing_files:
        print(f"Found existing training data in {training_dir}:")
        for file in existing_files:
            print(f"- {file}")
        print("Using existing training data instead of generating new data.")
        
        # Load the most recent training data file
        latest_file = max(existing_files, key=lambda x: os.path.getctime(os.path.join(training_dir, x)))
        filepath = os.path.join(training_dir, latest_file)
        
        with open(filepath, 'r') as f:
            training_data = json.load(f)
            
        # Convert to DSPy examples
        trainset = []
        for example in training_data:
            trainset.append(
                Example(
                    initial_prompt=example["initial_prompt"],
                    reasoning=example["reasoning"],
                    rewritten_prompt=example["rewritten_prompt"],
                    is_integral=example["is_integral"]
                ).with_inputs("initial_prompt")
            )
        
        print(f"Loaded {len(trainset)} examples from {latest_file}")
        return trainset
    
    print("No existing training data found. Generating new synthetic data...")
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    # First, generate diverse base prompts
    print("Generating diverse base prompts...")
    base_prompts = []
    
    # Generate prompts that can have polite words removed
    print("Generating prompts with removable polite words...")
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": """Generate a list of 50 unique prompts that contain polite words or phrases that can be safely removed.
Each prompt should be on a new line and start with a number (1-50).

Include different types of requests:
1. Help/explanation requests
2. Translation requests
3. Step-by-step guidance
4. Demonstration requests
5. Questions about concepts
6. Technical instructions
7. Creative writing prompts
8. Data analysis requests
9. Code review requests
10. Problem-solving requests

Make sure each prompt:
- Has clear polite words/phrases that can be removed
- Is meaningful and clear even without the polite words
- Is different from other prompts
- Covers a different topic or domain

Format each line as:
1. [prompt]
2. [prompt]
etc."""},
            {"role": "user", "content": "Generate 50 diverse prompts with removable polite words."}
        ],
        temperature=0.9
    )
    
    # Parse the response to get numbered prompts
    removable_prompts = []
    for line in response.choices[0].message.content.split('\n'):
        line = line.strip()
        # Remove bullet numbers and any leading/trailing whitespace
        if line and line[0].isdigit():
            # Remove the number and any following dots or spaces
            prompt = line.lstrip('0123456789.- ').strip()
            if prompt:
                removable_prompts.append({
                    "prompt": prompt,
                    "is_integral": False
                })
    
    print(f"Generated {len(removable_prompts)} prompts with removable polite words")
    base_prompts.extend([p["prompt"] for p in removable_prompts])
    
    # Generate prompts where polite words are integral to meaning
    print("Generating prompts with integral polite words...")
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": """Generate a list of 50 unique prompts where polite words or phrases are integral to the meaning and cannot be removed.
Each prompt should be on a new line and start with a number (1-50).

Examples:
- "Please be kind to others" (removing 'kind' changes meaning)
- "Translate this to german: It is important to be thankful for what we have (removing 'thankful' changes task)
- "how many s's are in thanks?" (removing 'thanks' would break the prompt)


Make sure each prompt:
- Has polite words that are essential to meaning
- Would change meaning if polite words are removed
- Is different from other prompts
- Covers a different context

Format each line as:
1. [prompt]
2. [prompt]
etc."""},
            {"role": "user", "content": "Generate 50 diverse prompts with integral polite words."}
        ],
        temperature=0.9
    )
    
    # Parse the response to get numbered prompts
    integral_prompts = []
    for line in response.choices[0].message.content.split('\n'):
        line = line.strip()
        # Remove bullet numbers and any leading/trailing whitespace
        if line and line[0].isdigit():
            # Remove the number and any following dots or spaces
            prompt = line.lstrip('0123456789.- ').strip()
            if prompt:
                integral_prompts.append({
                    "prompt": prompt,
                    "is_integral": True
                })
    
    print(f"Generated {len(integral_prompts)} prompts with integral polite words")
    base_prompts.extend([p["prompt"] for p in integral_prompts])
    print(integral_prompts)
    
    # Combine and shuffle prompts
    base_prompts = removable_prompts + integral_prompts
    random.shuffle(base_prompts)
    
    # Generate synthetic examples
    print(f"Generating {len(base_prompts)} synthetic examples...")
    synthetic_examples = []
    
    for prompt_data in base_prompts:
        prompt = prompt_data["prompt"]
        is_integral = prompt_data["is_integral"]
        
        try:
            # Call API for single prompt
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that rewrites prompts to remove polite words while maintaining the core meaning.
For each prompt:
1. Identify any polite words or phrases
2. Explain why they can or cannot be removed
3. Provide a rewritten version without the polite words if they can be removed, otherwise return the original prompt

You must format your response exactly like this:
Reasoning: [your explanation]
Rewritten: [the rewritten prompt]"""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            # Parse the response with better error handling
            content = response.choices[0].message.content.strip()
            
            # Try to parse the response
            try:
                if "Reasoning:" in content and "Rewritten:" in content:
                    reasoning = content.split("Rewritten:")[0].replace("Reasoning:", "").strip()
                    rewritten = content.split("Rewritten:")[1].strip()
                else:
                    # Fallback parsing if format is slightly different
                    parts = content.split("\n")
                    reasoning = ""
                    rewritten = ""
                    for part in parts:
                        if part.lower().startswith("reasoning:"):
                            reasoning = part.replace("Reasoning:", "").strip()
                        elif part.lower().startswith("rewritten:"):
                            rewritten = part.replace("Rewritten:", "").strip()
                    
                    if not reasoning or not rewritten:
                        raise ValueError("Could not parse response format")
                
                synthetic_examples.append({
                    "initial_prompt": prompt,
                    "reasoning": reasoning,
                    "rewritten_prompt": rewritten,
                    "is_integral": is_integral
                })
                
                print(f"Generated example {len(synthetic_examples)}/{len(base_prompts)}")
                
            except Exception as parse_error:
                print(f"Error parsing response for prompt: {prompt}")
                print(f"Parse error: {str(parse_error)}")
                print(f"Raw response: {content}")
                continue
            
        except Exception as e:
            print(f"Error generating example for prompt: {prompt}")
            print(f"Error: {str(e)}")
    
    if not synthetic_examples:
        raise Exception("Failed to generate any valid examples")
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"synthetic_training_data_{timestamp}.json"
    filepath = os.path.join(training_dir, filename)
    
    # Save to file
    with open(filepath, "w") as f:
        json.dump(synthetic_examples, f, indent=2)
    
    print(f"Generated {len(synthetic_examples)} synthetic examples")
    print(f"Training data saved to: {filepath}")
    
    # Convert to DSPy examples
    trainset = []
    for example in synthetic_examples:
        trainset.append(
            Example(
                initial_prompt=example["initial_prompt"],
                reasoning=example["reasoning"],
                rewritten_prompt=example["rewritten_prompt"],
                is_integral=example["is_integral"]
            ).with_inputs("initial_prompt")
        )
    
    return trainset

def optimize_prompt(trainset, num_train=20):
    """Optimize prompt using OpenAI's model."""
    print("Using OpenAI for prompt optimization...")
    
    # Take first num_train examples for optimization
    optimization_set = trainset[:num_train]
    print(f"Using {len(optimization_set)} examples for optimization")
    
    # Initialize the predictor
    predictor = PoliteWordRemoverPredictor()
    
    # Create the teleprompter with a custom metric that doesn't require pickling
    def optimization_metric(pred, gold, trace=None):
        """Metric for optimization that compares rewritten prompts."""
        return pred.rewritten_prompt.strip().lower() == gold.rewritten_prompt.strip().lower()
    
    teleprompter = BootstrapFewShot(
        metric=optimization_metric,  # Use simpler metric for optimization
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    # Create a function to get a fresh LM instance
    def get_lm():
        client = get_openai_client()
        return dspy.LM(
            model="gpt-4.1",
            client=client,
            temperature=0.7,
            max_tokens=1000,
            cache=False
        )
    
    # Optimize the prompt
    print("Optimizing prompt with GPT-4.1...")
    try:
        # Create a fresh LM instance for optimization
        openai_lm = get_lm()
        
        with dspy.context(lm=openai_lm):
            optimized_predictor = teleprompter.compile(predictor, trainset=optimization_set)
            
            # Print the optimized prompt
            print("\nOptimized Prompt:")
            print("----------------")
            print(optimized_predictor.predict.signature.instructions)
            print("\nFew-shot examples:")
            for demo in optimized_predictor.predict.demos:
                print(f"\nInitial Prompt: {demo.initial_prompt}")
                print(f"Reasoning: {demo.reasoning}")
                print(f"Rewritten Prompt: {demo.rewritten_prompt}")
            print("----------------\n")
            
            return optimized_predictor
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print("Falling back to basic predictor...")
        return predictor
    finally:
        # Clean up any resources
        try:
            if 'openai_lm' in locals():
                openai_lm.client.close()
        except:
            pass

def evaluate_prompt_rewriter(model_name, num_examples=80, use_optimized=True):
    """Evaluate the prompt rewriter on synthetic examples."""
    # Create results directory structure
    results_dir = os.path.join("results", "prompt_rewriter")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp and clean model name for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    # Generate or load training data
    trainset = generate_synthetic_data(num_examples=100)
    
    # Split data into optimization and evaluation sets
    optimization_set = trainset[:20]  # First 20 examples for optimization
    evaluation_set = trainset[20:20+num_examples]  # Next num_examples for evaluation
    
    print(f"\nUsing {len(optimization_set)} examples for optimization")
    print(f"Using {len(evaluation_set)} examples for evaluation")
    
    # Create a function to get a fresh local LM instance
    def get_local_lm():
        client = get_local_client()
        return dspy.LM(
            model=model_name,
            api_base='http://localhost:1234/v1',
            api_type='openai',
            api_key='not-needed',
            client=client,
            cache=False
        )
    
    # Initialize predictor (optimized or basic)
    if use_optimized:
        print("Using OpenAI-optimized prompt...")
        predictor = optimize_prompt(optimization_set)
    else:
        print("Using basic prompt...")
        predictor = PoliteWordRemoverPredictor()
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    latencies = []
    
    try:
        # Create a fresh LM instance for evaluation
        local_lm = get_local_lm()
        
        with dspy.context(lm=local_lm):
            for example in tqdm(evaluation_set):
                try:
                    start_time = time.time()
                    result = predictor(initial_prompt=example.initial_prompt)
                    end_time = time.time()
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                    # Evaluate correctness using custom metric
                    is_correct = custom_metric(
                        pred=result,
                        gold={
                            "initial_prompt": example.initial_prompt,
                            "is_integral": example.is_integral,
                            "rewritten_prompt": example.rewritten_prompt
                        }
                    )
                    
                    correct += int(is_correct)
                    total += 1
                    
                    results.append({
                        "initial_prompt": example.initial_prompt,
                        "expected_reasoning": example.reasoning,
                        "expected_rewritten": example.rewritten_prompt,
                        "model_reasoning": result.reasoning,
                        "model_rewritten": result.rewritten_prompt,
                        "is_integral": example.is_integral,
                        "is_correct": is_correct,
                        "latency": latency
                    })
                    
                    # Print progress
                    if len(results) % 10 == 0:
                        print(f"\nAccuracy so far: {correct/total:.2%}")
                        print(f"Median latency so far: {statistics.median(latencies):.2f}s")
                except Exception as e:
                    print(f"\nError processing prompt: {example.initial_prompt}")
                    print(f"Error: {str(e)}")
                    results.append({
                        "initial_prompt": example.initial_prompt,
                        "is_integral": example.is_integral,
                        "error": str(e),
                        "is_correct": False
                    })
                    total += 1
    finally:
        # Clean up any resources
        try:
            if 'local_lm' in locals():
                local_lm.client.close()
        except:
            pass
    
    # Calculate final metrics
    median_latency = statistics.median(latencies) if latencies else 0
    
    # Update results data with accuracy
    results_data = {
        "metadata": {
            "model_name": model_name,
            "timestamp": timestamp,
            "num_examples": total,
            "use_optimized_prompt": use_optimized,
            "optimized_by": "gpt-4.1" if use_optimized else "none"
        },
        "metrics": {
            "accuracy": correct/total,
            "correct": correct,
            "total": total,
            "median_latency": median_latency,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0
        },
        "results": results
    }
    
    # Save results with descriptive filename
    filename = f"prompt_rewriter_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    print(f"Total Examples: {total}")
    print(f"Accuracy: {correct/total:.2%}")
    print(f"Median Latency: {median_latency:.2f}s")
    print(f"Min Latency: {min(latencies):.2f}s" if latencies else "Min Latency: N/A")
    print(f"Max Latency: {max(latencies):.2f}s" if latencies else "Max Latency: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate local model on prompt rewriting.")
    parser.add_argument("--num_examples", type=int, default=80, help="Number of examples to use for evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Model name as recognized by LM Studio")
    parser.add_argument("--no_optimize", action="store_true", help="Disable prompt optimization")
    args = parser.parse_args()
    evaluate_prompt_rewriter(model_name=args.model_name, num_examples=args.num_examples, use_optimized=not args.no_optimize)