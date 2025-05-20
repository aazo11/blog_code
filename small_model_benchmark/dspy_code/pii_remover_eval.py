import dspy
from openai import OpenAI
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
import re

def get_openai_client():
    """Get a fresh OpenAI client instance."""
    return OpenAI()  # This will use OPENAI_API_KEY from environment

def get_local_client():
    """Get a fresh local client instance."""
    return OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="not-needed"
    )

class PIIRemoverSignature(dspy.Signature):
    """Removes PII (Personally Identifiable Information) from text while maintaining readability and meaning."""
    input_text: str = dspy.InputField()
    pii_elements: list = dspy.InputField(desc="List of PII elements found in the text, each with 'type' and 'value'")
    reasoning: str = dspy.OutputField(desc="Step by step reasoning about what PII was found and why it was removed")
    cleaned_text: str = dspy.OutputField(desc="The text with PII removed and replaced with appropriate placeholders")

class PIIRemoverPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PIIRemoverSignature)
        # Fallback predictor with more explicit instructions
        self.fallback_predict = dspy.Predict(
            PIIRemoverSignature,
            instructions="""You must provide both reasoning and cleaned text in your response.
First show your step-by-step reasoning about what PII was found, then on a new line write 'Cleaned: ' followed by the text with PII removed.
Example format:
Reasoning: [your explanation]
Cleaned: [the cleaned text]"""
        )
    
    def forward(self, input_text, pii_elements):
        """Process the input text and return the cleaned version."""
        try:
            return self.predict(input_text=input_text, pii_elements=pii_elements)
        except Exception as e:
            print("\nRetrying with more explicit instructions...")
            return self.fallback_predict(input_text=input_text, pii_elements=pii_elements)

def custom_metric(pred, gold, trace=None):
    """Custom metric for evaluating PII removal accuracy.
    
    For texts with PII:
    - Output should be different from input
    - Output should maintain readability
    - Output should not contain any of the identified PII elements
    - Output should use appropriate placeholders (format doesn't matter)
    
    For texts without PII:
    - Output should match input exactly
    """
    # Check if this is a text with PII
    has_pii = gold["has_pii"]
    input_text = gold["input_text"]
    pii_elements = gold["pii_elements"]
    
    if not has_pii:
        # For texts without PII, output should match input exactly
        return pred.cleaned_text.strip() == input_text.strip()
    else:
        # For texts with PII:
        # 1. Output should be different from input
        if pred.cleaned_text.strip() == input_text.strip():
            return False
            
        # 2. Check if any of the identified PII elements remain in the cleaned text
        for pii_element in pii_elements:
            if pii_element["value"] in pred.cleaned_text:
                return False
        
        # 3. Check if the cleaned text contains some form of redaction
        # Look for common redaction patterns like [REDACTED], [REMOVED], etc.
        redaction_patterns = [
            r'\[.*?\]',  # [anything in brackets]
            r'<.*?>',    # <anything in angle brackets>
            r'REDACTED',
            r'REMOVED',
            r'CONFIDENTIAL',
            r'PRIVATE',
            r'[Xx]+',    # XXXX or xxxx
            r'[*]+'      # ****
        ]
        
        has_redaction = any(re.search(pattern, pred.cleaned_text) for pattern in redaction_patterns)
        if not has_redaction:
            return False
            
        # 4. Output should maintain readability
        # This is a simple check - in practice you might want a more sophisticated semantic similarity check
        return True

def generate_synthetic_data(num_examples=100):
    """Generate synthetic training data and save it to file."""
    # Create training data directory
    training_dir = os.path.join("training_data", "pii_remover")
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
                    input_text=example["input_text"],
                    pii_elements=example["pii_elements"],
                    reasoning=example["reasoning"],
                    cleaned_text=example["cleaned_text"],
                    has_pii=example["has_pii"]
                ).with_inputs("input_text", "pii_elements")
            )
        
        print(f"Loaded {len(trainset)} examples from {latest_file}")
        return trainset
    
    print("No existing training data found. Generating new synthetic data...")
    
    # Initialize OpenAI client
    client = get_openai_client()
    
    # First, generate diverse base texts
    print("Generating diverse base texts...")
    base_texts = []
    
    # Generate texts with PII (75 samples to ensure 2/3 ratio)
    print("Generating texts with PII...")
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": """Generate a list of 75 unique text snippets that contain various types of PII.
Each text should be on a new line and start with a number (1-75).

Include different types of PII:
1. Names (first, last, full)
2. Email addresses
3. Phone numbers
4. Social Security numbers
5. Physical addresses
6. IP addresses
7. Credit card numbers
8. Dates of birth
9. Driver's license numbers
10. Passport numbers

Make sure each text:
- Contains at least one type of PII
- Is realistic and natural-sounding
- Is different from other texts
- Covers different contexts (emails, forms, documents, etc.)

Format each line as:
1. [text]
2. [text]
etc."""},
            {"role": "user", "content": "Generate 75 diverse texts with PII."}
        ],
        temperature=0.9
    )
    
    # Parse the response to get numbered texts
    pii_texts = []
    for line in response.choices[0].message.content.split('\n'):
        line = line.strip()
        # Remove bullet numbers and any leading/trailing whitespace
        if line and line[0].isdigit():
            # Remove the number and any following dots or spaces
            text = line.lstrip('0123456789.- ').strip()
            if text:
                pii_texts.append({
                    "text": text,
                    "has_pii": True
                })
    
    print(f"Generated {len(pii_texts)} texts with PII")
    base_texts.extend([p["text"] for p in pii_texts])
    
    # Generate texts without PII (25 samples to maintain 2/3 ratio)
    print("Generating texts without PII...")
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": """Generate a list of 25 unique text snippets that do NOT contain any PII.
Each text should be on a new line and start with a number (1-25).

Include different types of content:
1. General information
2. Instructions
3. Descriptions
4. Questions
5. Statements
6. Lists
7. Explanations
8. Observations
9. Comments
10. Notes

Make sure each text:
- Contains NO PII
- Is realistic and natural-sounding
- Is different from other texts
- Covers different contexts

Format each line as:
1. [text]
2. [text]
etc."""},
            {"role": "user", "content": "Generate 25 diverse texts without PII."}
        ],
        temperature=0.9
    )
    
    # Parse the response to get numbered texts
    non_pii_texts = []
    for line in response.choices[0].message.content.split('\n'):
        line = line.strip()
        # Remove bullet numbers and any leading/trailing whitespace
        if line and line[0].isdigit():
            # Remove the number and any following dots or spaces
            text = line.lstrip('0123456789.- ').strip()
            if text:
                non_pii_texts.append({
                    "text": text,
                    "has_pii": False
                })
    
    print(f"Generated {len(non_pii_texts)} texts without PII")
    base_texts.extend([p["text"] for p in non_pii_texts])
    
    # Combine and shuffle texts
    base_texts = pii_texts + non_pii_texts
    random.shuffle(base_texts)
    
    # Print the ratio of PII to non-PII texts
    total_texts = len(base_texts)
    pii_count = sum(1 for text in base_texts if text["has_pii"])
    non_pii_count = total_texts - pii_count
    print(f"\nFinal dataset composition:")
    print(f"Total texts: {total_texts}")
    print(f"Texts with PII: {pii_count} ({pii_count/total_texts:.1%})")
    print(f"Texts without PII: {non_pii_count} ({non_pii_count/total_texts:.1%})")
    
    # Generate synthetic examples
    print(f"\nGenerating {len(base_texts)} synthetic examples...")
    synthetic_examples = []
    
    for text_data in base_texts:
        text = text_data["text"]
        has_pii = text_data["has_pii"]
        
        try:
            # First, identify PII elements in the text
            if has_pii:
                pii_response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": """Identify all PII elements in the given text.
For each PII element, provide:
1. The type of PII (e.g., 'email', 'phone', 'name', 'address', 'credit_card', 'ssn', 'dob', 'driver_license', 'passport')
2. The exact value found in the text

Format your response as a JSON array of objects, each with 'type' and 'value' fields.
Example:
[
    {"type": "email", "value": "john.doe@example.com"},
    {"type": "phone", "value": "555-123-4567"}
]"""},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.7
                )
                
                try:
                    pii_elements = json.loads(pii_response.choices[0].message.content)
                except json.JSONDecodeError:
                    print(f"Error parsing PII elements for text: {text}")
                    continue
            else:
                pii_elements = []
            
            # Then, generate the cleaned version
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": """You are a helpful assistant that removes PII (Personally Identifiable Information) from text while maintaining readability.
For each text:
1. Identify any PII (names, emails, phone numbers, addresses, etc.)
2. Explain what PII was found and why it needs to be removed
3. Provide a cleaned version with PII replaced by appropriate placeholders

You must format your response exactly like this:
Reasoning: [your explanation]
Cleaned: [the cleaned text]"""},
                    {"role": "user", "content": text}
                ],
                temperature=0.7
            )
            
            # Parse the response with better error handling
            content = response.choices[0].message.content.strip()
            
            # Try to parse the response
            try:
                if "Reasoning:" in content and "Cleaned:" in content:
                    reasoning = content.split("Cleaned:")[0].replace("Reasoning:", "").strip()
                    cleaned = content.split("Cleaned:")[1].strip()
                else:
                    # Fallback parsing if format is slightly different
                    parts = content.split("\n")
                    reasoning = ""
                    cleaned = ""
                    for part in parts:
                        if part.lower().startswith("reasoning:"):
                            reasoning = part.replace("Reasoning:", "").strip()
                        elif part.lower().startswith("cleaned:"):
                            cleaned = part.replace("Cleaned:", "").strip()
                    
                    if not reasoning or not cleaned:
                        raise ValueError("Could not parse response format")
                
                synthetic_examples.append({
                    "input_text": text,
                    "pii_elements": pii_elements,
                    "reasoning": reasoning,
                    "cleaned_text": cleaned,
                    "has_pii": has_pii
                })
                
                print(f"Generated example {len(synthetic_examples)}/{len(base_texts)}")
                
            except Exception as parse_error:
                print(f"Error parsing response for text: {text}")
                print(f"Parse error: {str(parse_error)}")
                print(f"Raw response: {content}")
                continue
            
        except Exception as e:
            print(f"Error generating example for text: {text}")
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
                input_text=example["input_text"],
                pii_elements=example["pii_elements"],
                reasoning=example["reasoning"],
                cleaned_text=example["cleaned_text"],
                has_pii=example["has_pii"]
            ).with_inputs("input_text", "pii_elements")
        )
    
    return trainset

def optimize_prompt(trainset, num_train=20):
    """Optimize prompt using OpenAI's model."""
    print("Using OpenAI for prompt optimization...")
    
    # Take first num_train examples for optimization
    optimization_set = trainset[:num_train]
    print(f"Using {len(optimization_set)} examples for optimization")
    
    # Initialize the predictor
    predictor = PIIRemoverPredictor()
    
    # Create the teleprompter with a custom metric that doesn't require pickling
    def optimization_metric(pred, gold, trace=None):
        """Metric for optimization that compares cleaned texts."""
        return pred.cleaned_text.strip().lower() == gold.cleaned_text.strip().lower()
    
    teleprompter = BootstrapFewShot(
        metric=optimization_metric,  # Use simpler metric for optimization
        max_bootstrapped_demos=3,
        max_labeled_demos=3
    )
    
    # Create a function to get a fresh LM instance
    def get_lm():
        client = get_openai_client()
        return dspy.LM(
            model="gpt-4-turbo-preview",
            client=client,
            temperature=0.7,
            max_tokens=1000,
            cache=False
        )
    
    # Optimize the prompt
    print("Optimizing prompt with GPT-4...")
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
                print(f"\nInput Text: {demo.input_text}")
                print(f"Reasoning: {demo.reasoning}")
                print(f"Cleaned Text: {demo.cleaned_text}")
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

def evaluate_pii_remover(model_name, num_examples=80, use_optimized=True):
    """Evaluate the PII remover on synthetic examples."""
    # Create results directory structure
    results_dir = os.path.join("results", "pii_remover")
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
        predictor = PIIRemoverPredictor()
    
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
                    result = predictor(input_text=example.input_text, pii_elements=example.pii_elements)
                    end_time = time.time()
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                    # Evaluate correctness using custom metric
                    is_correct = custom_metric(
                        pred=result,
                        gold={
                            "input_text": example.input_text,
                            "pii_elements": example.pii_elements,
                            "has_pii": example.has_pii,
                            "cleaned_text": example.cleaned_text
                        }
                    )
                    
                    correct += int(is_correct)
                    total += 1
                    
                    results.append({
                        "input_text": example.input_text,
                        "pii_elements": example.pii_elements,
                        "expected_reasoning": example.reasoning,
                        "expected_cleaned": example.cleaned_text,
                        "model_reasoning": result.reasoning,
                        "model_cleaned": result.cleaned_text,
                        "has_pii": example.has_pii,
                        "is_correct": is_correct,
                        "latency": latency
                    })
                    
                    # Print progress
                    if len(results) % 10 == 0:
                        print(f"\nAccuracy so far: {correct/total:.2%}")
                        print(f"Median latency so far: {statistics.median(latencies):.2f}s")
                except Exception as e:
                    print(f"\nError processing text: {example.input_text}")
                    print(f"Error: {str(e)}")
                    results.append({
                        "input_text": example.input_text,
                        "pii_elements": example.pii_elements,
                        "has_pii": example.has_pii,
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
    filename = f"pii_remover_{clean_model_name}_{timestamp}.json"
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
    parser = argparse.ArgumentParser(description="Evaluate local model on PII removal.")
    parser.add_argument("--num_examples", type=int, default=80, help="Number of examples to use for evaluation")
    parser.add_argument("--model_name", type=str, required=True, help="Model name as recognized by LM Studio")
    parser.add_argument("--no_optimize", action="store_true", help="Disable prompt optimization")
    args = parser.parse_args()
    evaluate_pii_remover(model_name=args.model_name, num_examples=args.num_examples, use_optimized=not args.no_optimize) 