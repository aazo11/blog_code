import dspy
from openai import OpenAI
from datasets import load_dataset
import json
from tqdm import tqdm
import argparse
from dspy.teleprompt import BootstrapFewShot
import os
from datetime import datetime
import shutil
import time
import statistics

# Configure DSPy with LM Studio for local model
local_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

# Configure OpenAI client for prompt optimization
openai_client = OpenAI()  # This will use OPENAI_API_KEY from environment

class GSM8KSignature(dspy.Signature):
    """Solve a math word problem step by step."""
    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step by step reasoning and calculations")
    answer: str = dspy.OutputField(desc="The final numerical answer only")

class GSM8KPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(GSM8KSignature, output_format="text")
        # Fallback predictor with more explicit instructions
        self.fallback_predict = dspy.Predict(
            GSM8KSignature,
            output_format="text",
            instructions="""You must provide both reasoning and answer in your response.
First show your step-by-step reasoning, then on a new line write 'Answer: ' followed by the final number."""
        )
    
    def forward(self, question):
        try:
            return self.predict(question=question)
        except ValueError as e:
            if "Expected" in str(e) and "but got" in str(e):
                print("\nRetrying with more explicit instructions...")
                return self.fallback_predict(question=question)
            raise e

def optimize_prompt(num_train=10):
    """Optimize prompt using OpenAI's model."""
    print("Using OpenAI for prompt optimization...")
    
    # Configure DSPy with OpenAI
    dspy.configure(
        lm=dspy.LM(
            model="gpt-4-turbo-preview",  # Using GPT-4 for better optimization
            api_type='openai',
            client=openai_client
        )
    )
    
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"].select(range(num_train))
    
    # Create training examples
    trainset = []
    for example in train_data:
        question = example["question"]
        # Split the answer into reasoning and final answer
        full_answer = example["answer"]
        reasoning = full_answer.split("####")[0].strip()
        correct_answer = full_answer.split("####")[1].strip()
        
        # Create example with proper input/output structure
        trainset.append(
            dspy.Example(
                question=question,
                reasoning=reasoning,
                answer=correct_answer
            ).with_inputs("question")
        )
    
    # Initialize the predictor
    predictor = GSM8KPredictor()
    
    # Instead of using BootstrapFewShot, we'll use a simpler approach
    # that doesn't require pickling the LM client
    print("Optimizing prompt with GPT-4...")
    
    # Create a prompt that includes a few examples
    examples = trainset[:3]  # Use first 3 examples
    example_text = "\n\n".join([
        f"Question: {ex.question}\nReasoning: {ex.reasoning}\nAnswer: {ex.answer}"
        for ex in examples
    ])
    
    # Create an optimized prompt using GPT-4
    optimization_prompt = f"""Based on these examples of math word problems and their solutions:

{example_text}

Create a clear and effective prompt that will help a language model solve similar math word problems.
The prompt should:
1. Guide the model to show step-by-step reasoning
2. Ensure the model provides a clear final answer
3. Help the model avoid common mistakes
4. Be concise but comprehensive

Format your response as a single prompt that can be used to solve similar problems."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a prompt engineering expert."},
                {"role": "user", "content": optimization_prompt}
            ],
            temperature=0.7
        )
        
        optimized_instructions = response.choices[0].message.content.strip()
        
        # Create a new predictor with the optimized instructions
        optimized_predictor = GSM8KPredictor()
        optimized_predictor.predict = dspy.Predict(
            GSM8KSignature,
            instructions=optimized_instructions
        )
        
        # Print the optimized prompt
        print("\nOptimized Prompt:")
        print("----------------")
        print(optimized_instructions)
        print("\nExample solutions:")
        for ex in examples:
            print(f"\nQuestion: {ex.question}")
            print(f"Reasoning: {ex.reasoning}")
            print(f"Answer: {ex.answer}")
        print("----------------\n")
        
        return optimized_predictor
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print("Falling back to basic predictor...")
        return predictor

def evaluate_gsm8k(model_name, num_questions=None, use_optimized=True):
    # Create results directory structure
    results_dir = os.path.join("results", "gsm8k")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp and clean model name for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace("/", "_").replace("\\", "_")
    
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset["test"]
    if num_questions is not None:
        test_data = test_data.select(range(min(num_questions, len(test_data))))
    
    # Initialize predictor (optimized or basic)
    if use_optimized:
        print("Using OpenAI-optimized prompt...")
        predictor = optimize_prompt()
    else:
        print("Using basic prompt...")
        predictor = GSM8KPredictor()
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    latencies = []
    
    for example in tqdm(test_data):
        question = example["question"]
        correct_answer = example["answer"].split("#### ")[1].strip()
        
        # Create a new LM client for each evaluation to avoid pickling issues
        dspy.configure(
            lm=dspy.LM(
                model=model_name,
                api_base='http://localhost:1234/v1',
                api_type='openai',
                api_key='not-needed',
                client=OpenAI(
                    base_url="http://localhost:1234/v1",
                    api_key="not-needed"
                ),
                cache=False  # Disable caching to avoid pickling
            )
        )
        
        # Get model's answer with retry logic and latency tracking
        try:
            start_time = time.time()
            result = predictor(question=question)
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            model_answer = result.answer.strip()
            # Remove any non-numeric characters except decimal points and negative signs
            model_answer = ''.join(c for c in model_answer if c.isdigit() or c in '.-')
            
            # Check if answer is correct
            is_correct = model_answer == correct_answer
            correct += int(is_correct)
            total += 1
            
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "model_answer": model_answer,
                "model_reasoning": result.reasoning,
                "is_correct": is_correct,
                "latency": latency
            })
            
            # Print progress
            if total % 10 == 0:
                print(f"\nAccuracy so far: {correct/total:.2%}")
                print(f"Median latency so far: {statistics.median(latencies):.2f}s")
        except Exception as e:
            print(f"\nError processing question: {question}")
            print(f"Error: {str(e)}")
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "error": str(e),
                "is_correct": False
            })
            total += 1
    
    # Calculate final metrics
    median_latency = statistics.median(latencies) if latencies else 0
    
    # Prepare results data with metadata
    results_data = {
        "metadata": {
            "model_name": model_name,
            "timestamp": timestamp,
            "num_questions": total,
            "use_optimized_prompt": use_optimized,
            "optimized_by": "gpt-4-turbo-preview" if use_optimized else "none"
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
    filename = f"gsm8k_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    print(f"Final Accuracy: {correct/total:.2%}")
    print(f"Correct: {correct}/{total}")
    print(f"Median Latency: {median_latency:.2f}s")
    print(f"Min Latency: {min(latencies):.2f}s" if latencies else "Min Latency: N/A")
    print(f"Max Latency: {max(latencies):.2f}s" if latencies else "Max Latency: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate local model on GSM8K.")
    parser.add_argument("--num_questions", type=int, default=None, help="Number of questions to run (default: all)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name as recognized by LM Studio")
    parser.add_argument("--no_optimize", action="store_true", help="Disable prompt optimization")
    args = parser.parse_args()
    evaluate_gsm8k(model_name=args.model_name, num_questions=args.num_questions, use_optimized=not args.no_optimize) 