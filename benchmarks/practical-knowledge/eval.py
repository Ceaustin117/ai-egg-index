#!/usr/bin/env python3
"""
Practical Knowledge Index Evaluator

Evaluates free tier LLMs on real-world practical knowledge questions.
Uses LLM-as-judge for scoring with rubric-based evaluation.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")
    sys.exit(1)


class PracticalKnowledgeEvaluator:
    """Evaluates LLM responses on practical knowledge questions."""

    def __init__(self, judge_model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.judge_model = judge_model
        self.questions = self._load_questions()

    def _load_questions(self) -> dict:
        """Load questions from JSON file."""
        questions_path = Path(__file__).parent / "questions.json"
        with open(questions_path) as f:
            return json.load(f)

    def _call_llm(self, model: str, prompt: str, max_tokens: int = 1024) -> str:
        """Call an LLM and return the response."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _judge_response(self, question: dict, response: str) -> dict:
        """Use LLM-as-judge to score a response."""
        judge_prompt = f"""You are evaluating an AI assistant's response to a practical knowledge question.

QUESTION: {question['question']}

EXPECTED TOPICS TO COVER: {', '.join(question['expected_topics'])}

VERIFICATION SOURCE: {question['verification_source']}

AI RESPONSE:
{response}

Score the response on three criteria (0.0 to 1.0 each):

1. FACTUAL_ACCURACY (weight: 50%): Is the information correct and verifiable?
   - 1.0 = All facts are accurate
   - 0.5 = Some facts correct, some errors or outdated
   - 0.0 = Major factual errors

2. COMPLETENESS (weight: 30%): Does it cover the expected topics?
   - 1.0 = Covers all expected topics thoroughly
   - 0.5 = Covers some topics, misses others
   - 0.0 = Misses most expected topics

3. RECENCY_AWARENESS (weight: 20%): Does it acknowledge current year or knowledge cutoff?
   - 1.0 = Correctly states current info or acknowledges cutoff
   - 0.5 = Vague about timeframe
   - 0.0 = Presents outdated info as current without disclaimer

Respond in JSON format:
{{
  "factual_accuracy": 0.0-1.0,
  "completeness": 0.0-1.0,
  "recency_awareness": 0.0-1.0,
  "explanation": "Brief explanation of scores"
}}"""

        judge_response = self._call_llm(self.judge_model, judge_prompt, max_tokens=512)

        try:
            # Extract JSON from response
            json_start = judge_response.find('{')
            json_end = judge_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                scores = json.loads(judge_response[json_start:json_end])
                return scores
        except json.JSONDecodeError:
            pass

        # Default scores if parsing fails
        return {
            "factual_accuracy": 0.5,
            "completeness": 0.5,
            "recency_awareness": 0.5,
            "explanation": "Could not parse judge response"
        }

    def evaluate_model(self, model: str, limit: int = None) -> dict:
        """Evaluate a model on all practical knowledge questions."""
        results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "overall_score": 0.0
        }

        all_scores = []

        for category, questions in self.questions["categories"].items():
            category_results = []

            for i, question in enumerate(questions):
                if limit and i >= limit:
                    break

                print(f"  Evaluating: {question['id']}")

                # Get model response
                response = self._call_llm(model, question["question"])

                # Judge the response
                scores = self._judge_response(question, response)

                # Calculate weighted score
                weighted_score = (
                    scores.get("factual_accuracy", 0) * 0.5 +
                    scores.get("completeness", 0) * 0.3 +
                    scores.get("recency_awareness", 0) * 0.2
                )

                category_results.append({
                    "question_id": question["id"],
                    "response": response[:500] + "..." if len(response) > 500 else response,
                    "scores": scores,
                    "weighted_score": round(weighted_score, 3)
                })

                all_scores.append(weighted_score)

            if category_results:
                category_avg = sum(r["weighted_score"] for r in category_results) / len(category_results)
                results["categories"][category] = {
                    "questions": category_results,
                    "average_score": round(category_avg, 3)
                }

        # Calculate overall score
        if all_scores:
            results["overall_score"] = round(sum(all_scores) / len(all_scores), 3)

        return results

    def save_results(self, results: dict, output_dir: str = None) -> str:
        """Save evaluation results to JSON file."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "results" / datetime.now().strftime("%Y-%m-%d")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize model name for filename
        model_name = results["model"].replace("/", "-").replace(":", "-")
        output_path = output_dir / f"practical-knowledge-{model_name}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return str(output_path)


def main():
    """Run the practical knowledge benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM on practical knowledge questions")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model to evaluate")
    parser.add_argument("--limit", type=int, help="Limit questions per category")
    parser.add_argument("--output", help="Output directory for results")

    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Evaluating model: {args.model}")
    print("-" * 50)

    evaluator = PracticalKnowledgeEvaluator()
    results = evaluator.evaluate_model(args.model, limit=args.limit)

    output_path = evaluator.save_results(results, args.output)

    print("-" * 50)
    print(f"Overall Score: {results['overall_score']}")
    print(f"Results saved to: {output_path}")

    # Print category breakdown
    print("\nCategory Scores:")
    for category, data in results["categories"].items():
        print(f"  {category}: {data['average_score']}")


if __name__ == "__main__":
    main()
