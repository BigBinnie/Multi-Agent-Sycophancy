from typing import Optional, Dict, List, Callable, Any
from datasets import load_dataset
import json
import random
import os

async def load_gsm8k_questions(split: str = "train", num_samples: int = 1) -> List[Dict]:
    """
    Load questions and answers from GSM8k dataset
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    questions_data = []
    
    for i, item in enumerate(dataset):
        questions_data.append({
            'question': item['question'],
            'answer': float(item['answer'].split('####')[1].replace(",", "").strip())  # Extract numerical answer
        })   
    questions_data = random.sample(questions_data, num_samples)
    print(len(questions_data))
    return questions_data

async def load_gsm8k_difficult_questions(split: str = "train", num_samples: int = 1) -> List[Dict]:
    """
    Load questions and answers from GSM8k dataset
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    questions_data = []
    
    for i, item in enumerate(dataset):
        questions_data.append({
            'question': item['question'],
            'answer': float(item['answer'].split('####')[1].replace(",", "").strip())  # Extract numerical answer
        })
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the dataset file
    dataset_path = os.path.join(script_dir, "dataset", "gsm8k_difficult.json")
    with open(dataset_path, "r") as f:
        difficult_questions = json.load(f)
    difficult_question_map = {}
    for w in difficult_questions:
        difficult_question_map[w["question"]] = w
    questions_data = [ q for q in questions_data if q["question"] in difficult_question_map]    
    
    questions_data = random.sample(questions_data, num_samples)
    print(len(questions_data))
    return questions_data


async def load_mmlu_pro(split: str = "test", num_samples: int = 1, category: Optional[str] = None) -> List[Dict]:
    """
    Load questions and answers from MMLU Pro dataset
    
    The dataset contains multiple-choice questions on various professional and academic topics.
    Each question has a list of possible answers, and one correct answer.
    
    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        num_samples: Number of samples to return
        category: Optional category to filter questions by (e.g., 'biology', 'physics')
    """
    # Load the dataset from HuggingFace
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    questions_data = []
    
    for item in dataset:
        question = item['question']
        
        # Handle options based on the format
        options = {}
        
        # Check if options is in the new format (list)
        if 'options' in item and isinstance(item['options'], list):
            for i, option_text in enumerate(item['options']):
                # Convert index to letter (0->A, 1->B, etc.)
                letter = chr(65 + i)  # 65 is ASCII for 'A'
                options[letter] = option_text
        else:
            # Handle the old format where options are individual fields (A, B, C, D)
            option_keys = ['A', 'B', 'C', 'D']
            for key in option_keys:
                if key in item:
                    options[key] = item[key]
        
        # Get the correct answer
        if 'answer_index' in item:
            # Convert index to letter (0->A, 1->B, etc.)
            correct_letter = chr(65 + item['answer_index'])
        else:
            correct_letter = item['answer']
        
        correct_answer = options.get(correct_letter, "")
        
        questions_data.append({
            'question': question,
            'options': options,
            'answer': correct_letter,
            'answer_content': correct_answer,
            'category': item.get('category', ''),
            'question_id': item.get('question_id', ''),
            'src': item.get('src', '')
        })
    
    # Filter by category if specified
    if category:
        filtered_data = [q for q in questions_data if q['category'].lower() == category.lower()]
        if not filtered_data:
            print(f"Warning: No questions found for category '{category}'. Using all questions.")
        else:
            questions_data = filtered_data
            print(f"Filtered to {len(questions_data)} questions in category '{category}'")
    
    if num_samples < len(questions_data):
        questions_data = random.sample(questions_data, num_samples)
    return questions_data

async def load_bigbenchhard(subset: str = "date_understanding", split: str = "train", num_samples: int = 1) -> List[Dict]:
    """
    Load questions and answers from BigBenchHard dataset
    
    The dataset contains multiple-choice questions on various topics.
    Each question has a list of possible answers, and one correct answer.
    
    Args:
        subset: The subset of BigBenchHard to load (e.g., 'date_understanding', 'disambiguation_qa', etc.)
        split: Dataset split to use ('train', 'test')
        num_samples: Number of samples to return
    """
    dataset = load_dataset("maveriq/bigbenchhard", subset, split=split)
    questions_data = []
    
    for i, item in enumerate(dataset):
        # The input contains both the question and options
        input_text = item['input']
        
        # Split the input into question and options
        parts = input_text.split("Options:")
        question = parts[0].strip()
        
        # Parse the options
        options_text = parts[1].strip() if len(parts) > 1 else ""
        options = {}
        option_lines = options_text.split("\n")
        for line in option_lines:
            if line.strip():
                # Extract option letter and text (e.g., "(A) 12/11/1937" -> "A": "12/11/1937")
                option_match = line.strip()
                option_letter = option_match[1:2]  # Extract the letter inside parentheses
                option_text = option_match[4:].strip()  # Extract the text after the letter
                options[option_letter] = option_text
        
        # Get the correct answer
        target = item['target'].strip()  # e.g., "(B)"
        correct_letter = target[1:2]  # Extract the letter inside parentheses
        correct_answer = options.get(correct_letter, "")
        
        questions_data.append({
            'question': question,
            'options': options,
            'answer': correct_letter,
            'answer_content': correct_answer
        })
    
    if num_samples < len(questions_data):
        questions_data = random.sample(questions_data, num_samples)
    print(f"Loaded {len(questions_data)} samples from BigBenchHard {subset}")
    return questions_data

async def load_commonsenseqa(split: str = "validation", num_samples: int = 1) -> List[Dict]:
    """
    Load questions and answers from CommonsenseQA dataset
    
    The dataset contains multiple-choice questions that require commonsense knowledge.
    Each question has 5 possible answers (A, B, C, D, E), and one correct answer.
    
    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        num_samples: Number of samples to return
    """
    dataset = load_dataset("tau/commonsense_qa", split=split)
    questions_data = []
    
    empty_answer_count = 0
    empty_content_count = 0
    
    for i, item in enumerate(dataset):
        question = item['question']
        
        # Extract options from the choices structure
        options = {}
        choice_labels = item['choices']['label']  # ['A', 'B', 'C', 'D', 'E']
        choice_texts = item['choices']['text']    # List of option texts
        
        for label, text in zip(choice_labels, choice_texts):
            options[label] = text
        
        # Get the correct answer
        correct_letter = item['answerKey']
        correct_answer = options.get(correct_letter, "")
        
        # Debug logging for empty fields
        if not correct_letter or correct_letter.strip() == '':
            empty_answer_count += 1
            print(f"[WARNING] Empty answerKey found at index {i}: {item}")
        
        if not correct_answer or correct_answer.strip() == '':
            empty_content_count += 1
            print(f"[WARNING] Empty answer_content found at index {i}:")
            print(f"  Question: {question}")
            print(f"  AnswerKey: '{correct_letter}'")
            print(f"  Options: {options}")
            print(f"  Answer content: '{correct_answer}'")
        
        questions_data.append({
            'question': question,
            'options': options,
            'answer': correct_letter,
            'answer_content': correct_answer,
            'question_concept': item.get('question_concept', ''),
            'id': item.get('id', '')
        })
    
    if num_samples < len(questions_data):
        questions_data = random.sample(questions_data, num_samples)
    
    # Final validation
    final_empty_answer = sum(1 for q in questions_data if not q['answer'] or q['answer'].strip() == '')
    final_empty_content = sum(1 for q in questions_data if not q['answer_content'] or q['answer_content'].strip() == '')
    
    print(f"Loaded {len(questions_data)} samples from CommonsenseQA {split}")
    if empty_answer_count > 0:
        print(f"[WARNING] Found {empty_answer_count} items with empty answerKey in dataset")
    if empty_content_count > 0:
        print(f"[WARNING] Found {empty_content_count} items with empty answer_content in dataset")
    if final_empty_answer > 0:
        print(f"[WARNING] Final result has {final_empty_answer} items with empty answer field")
    if final_empty_content > 0:
        print(f"[WARNING] Final result has {final_empty_content} items with empty answer_content field")
    
    return questions_data

async def load_ai2_arc_challenge(split: str = "test", num_samples: int = 1) -> List[Dict]:
    """
    Load questions and answers from AI2 ARC Challenge dataset
    
    The dataset contains multiple-choice science questions that require reasoning.
    Each question has 4 possible answers (A, B, C, D), and one correct answer.
    
    Args:
        split: Dataset split to use ('train', 'validation', 'test')
        num_samples: Number of samples to return
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    questions_data = []
    
    empty_answer_count = 0
    empty_content_count = 0
    
    for i, item in enumerate(dataset):
        question = item['question']
        
        # Extract options from the choices structure
        options = {}
        choice_labels = item['choices']['label']  # ['A', 'B', 'C', 'D']
        choice_texts = item['choices']['text']    # List of option texts
        
        for label, text in zip(choice_labels, choice_texts):
            options[label] = text
        
        # Get the correct answer
        correct_letter = item['answerKey']
        correct_answer = options.get(correct_letter, "")
        
        # Debug logging for empty fields
        if not correct_letter or correct_letter.strip() == '':
            empty_answer_count += 1
            print(f"[WARNING] Empty answerKey found at index {i}: {item}")
        
        if not correct_answer or correct_answer.strip() == '':
            empty_content_count += 1
            print(f"[WARNING] Empty answer_content found at index {i}:")
            print(f"  Question: {question}")
            print(f"  AnswerKey: '{correct_letter}'")
            print(f"  Options: {options}")
            print(f"  Answer content: '{correct_answer}'")
        
        questions_data.append({
            'question': question,
            'options': options,
            'answer': correct_letter,
            'answer_content': correct_answer,
            'id': item.get('id', '')
        })
    
    if num_samples < len(questions_data):
        questions_data = random.sample(questions_data, num_samples)
    
    # Final validation
    final_empty_answer = sum(1 for q in questions_data if not q['answer'] or q['answer'].strip() == '')
    final_empty_content = sum(1 for q in questions_data if not q['answer_content'] or q['answer_content'].strip() == '')
    
    print(f"Loaded {len(questions_data)} samples from AI2 ARC Challenge {split}")
    if empty_answer_count > 0:
        print(f"[WARNING] Found {empty_answer_count} items with empty answerKey in dataset")
    if empty_content_count > 0:
        print(f"[WARNING] Found {empty_content_count} items with empty answer_content in dataset")
    if final_empty_answer > 0:
        print(f"[WARNING] Final result has {final_empty_answer} items with empty answer field")
    if final_empty_content > 0:
        print(f"[WARNING] Final result has {final_empty_content} items with empty answer_content field")
    
    return questions_data

def apply_partition_config(questions_data: List[Dict], partition_config: Dict) -> List[Dict]:
    """
    Apply data partitioning configuration to the dataset
    
    Args:
        questions_data: List of question dictionaries
        partition_config: Configuration for data partitioning with the following keys:
            - partition_type: Type of partitioning ('index', 'percentage', 'random')
            - start_idx: Starting index for 'index' partitioning
            - end_idx: Ending index for 'index' partitioning
            - percentage: Percentage of data to use for 'percentage' partitioning
            - seed: Random seed for 'random' partitioning
            
    Returns:
        Partitioned list of question dictionaries
    """
    partition_type = partition_config.get('partition_type', 'index')
    
    if partition_type == 'index':
        # Partition by index range
        start_idx = partition_config.get('start_idx', 0)
        end_idx = partition_config.get('end_idx', len(questions_data))
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(questions_data)))
        end_idx = max(start_idx, min(end_idx, len(questions_data)))
        
        return questions_data[start_idx:end_idx]
        
    elif partition_type == 'percentage':
        # Partition by percentage
        percentage = partition_config.get('percentage', 100)
        percentage = max(0, min(percentage, 100))  # Ensure percentage is between 0 and 100
        
        # Calculate number of samples
        num_samples = int(len(questions_data) * percentage / 100)
        
        # Set seed if provided
        if 'seed' in partition_config:
            random.seed(partition_config['seed'])
            
        # Sample the data
        return random.sample(questions_data, num_samples) if num_samples > 0 else []
        
    elif partition_type == 'random':
        # Partition by random selection
        num_samples = partition_config.get('num_samples', len(questions_data))
        num_samples = max(0, min(num_samples, len(questions_data)))
        
        # Set seed if provided
        if 'seed' in partition_config:
            random.seed(partition_config['seed'])
            
        # Sample the data
        return random.sample(questions_data, num_samples) if num_samples > 0 else []
        
    else:
        # Unknown partition type, return all data
        print(f"Warning: Unknown partition type '{partition_type}'. Using all data.")
        return questions_data


async def get_dataset(dataset_name: str, subset: str, split: str = "train", num_samples: int = 1, 
                     category: Optional[str] = None, partition_config: Optional[Dict] = None) -> List[Dict]:
    """
    Load questions and answers from a specified dataset
    
    Args:
        dataset_name: Name of the dataset to load ('gsm8k', 'gsm8k_difficult', 'bigbenchhard', 'mmlu_pro', 'commonsenseqa', 'ai2_arc_challenge')
        subset: Subset of the dataset to load (used for bigbenchhard)
        split: Dataset split to use ('train', 'validation', 'test')
        num_samples: Number of samples to return
        category: Optional category to filter questions by (for mmlu_pro)
        partition_config: Optional configuration for data partitioning with the following keys:
            - partition_type: Type of partitioning ('index', 'percentage', 'random')
            - start_idx: Starting index for 'index' partitioning
            - end_idx: Ending index for 'index' partitioning
            - percentage: Percentage of data to use for 'percentage' partitioning
            - seed: Random seed for 'random' partitioning
        
    Returns:
        List of dictionaries containing questions and answers
    """
    # Handle datasets with configurable subsets
    if dataset_name == "bigbenchhard":
        questions_data = await load_bigbenchhard(subset=subset, split=split, num_samples=num_samples)  # Get all samples first
    # Handle other datasets
    else:
        dataset_loaders = {
            "gsm8k": load_gsm8k_questions,
            "gsm8k_difficult": load_gsm8k_difficult_questions,
            "mmlu_pro":  load_mmlu_pro,
            "commonsenseqa": load_commonsenseqa,
            "ai2_arc_challenge": load_ai2_arc_challenge
        }
        
        if dataset_name not in dataset_loaders:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {list(dataset_loaders.keys()) + ['bigbenchhard']}")
        
        # Pass category parameter for mmlu_pro dataset
        if dataset_name == "mmlu_pro":
            questions_data = await load_mmlu_pro(split=split, num_samples=num_samples, category=category)  # Get all samples first
        else:
            questions_data = await dataset_loaders[dataset_name](split=split, num_samples=num_samples)  # Get all samples first
    
    if partition_config:
        questions_data = apply_partition_config(questions_data, partition_config)
        print(f"Applied partition config: {partition_config}")
        print(f"Resulting dataset size: {len(questions_data)}")
    
    return questions_data
