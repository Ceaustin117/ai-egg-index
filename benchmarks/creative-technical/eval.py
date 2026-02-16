#!/usr/bin/env python3
"""
Creative+Technical Benchmark Evaluator

Evaluates free tier LLMs on tasks requiring both coding skills and creative constraints.
Uses code execution for technical scoring and LLM-as-judge for creative scoring.
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:
    print("Please install groq: pip install groq")
    sys.exit(1)


class CreativeTechnicalEvaluator:
    """Evaluates LLM responses on creative+technical challenges."""

    def __init__(self, judge_model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.judge_model = judge_model
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load prompts from JSON file."""
        prompts_path = Path(__file__).parent / "prompts.json"
        with open(prompts_path) as f:
            return json.load(f)

    def _call_llm(self, model: str, prompt: str, max_tokens: int = 2048) -> str:
        """Call an LLM and return the response."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"

    def _extract_code(self, response: str, language: str = "python") -> str:
        """Extract code block from LLM response."""
        # Look for markdown code blocks
        markers = [f"```{language}", "```py", "```python", "```javascript", "```js", "```"]

        code = ""
        in_block = False
        current_marker = None

        for line in response.split('\n'):
            if not in_block:
                for marker in markers:
                    if line.strip().startswith(marker):
                        in_block = True
                        current_marker = "```"
                        break
            elif line.strip() == "```":
                break
            elif in_block:
                code += line + "\n"

        # If no code block found, try to extract any code-like content
        if not code.strip():
            lines = []
            for line in response.split('\n'):
                # Heuristic: lines that look like code
                if any(kw in line for kw in ['def ', 'class ', 'import ', 'function ', 'const ', 'let ', 'var ']):
                    lines.append(line)
                elif lines and (line.startswith('    ') or line.startswith('\t') or line.strip() == ''):
                    lines.append(line)
            code = '\n'.join(lines)

        return code.strip()

    def _test_code_execution(self, code: str, language: str = "python") -> dict:
        """Test if code executes without errors."""
        if not code:
            return {"runs": False, "error": "No code extracted", "output": ""}

        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py' if language == 'python' else '.js',
                delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            if language == "python":
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
            else:
                result = subprocess.run(
                    ["node", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

            os.unlink(temp_path)

            return {
                "runs": result.returncode == 0,
                "output": result.stdout[:1000] if result.stdout else "",
                "error": result.stderr[:500] if result.stderr else ""
            }

        except subprocess.TimeoutExpired:
            return {"runs": False, "error": "Execution timeout", "output": ""}
        except FileNotFoundError:
            return {"runs": False, "error": "Runtime not found", "output": ""}
        except Exception as e:
            return {"runs": False, "error": str(e), "output": ""}

    def _judge_creative(self, prompt_data: dict, response: str, code_result: dict) -> dict:
        """Use LLM-as-judge to score creative and requirements aspects."""
        judge_prompt = f"""You are evaluating an AI's response to a creative+technical challenge.

ORIGINAL PROMPT:
{prompt_data['prompt']}

SCORING CRITERIA:
- code_works (40%): {prompt_data['scoring']['code_works']['description']}
- meets_requirements (30%): {prompt_data['scoring']['meets_requirements']['description']}
- creative_output (30%): {prompt_data['scoring']['creative_output']['description']}

AI RESPONSE:
{response[:2000]}

CODE EXECUTION RESULT:
- Runs without error: {code_result['runs']}
- Output: {code_result['output'][:500]}
- Error: {code_result['error'][:200]}

Score each criterion from 0.0 to 1.0. Be strict but fair.

Respond in JSON format:
{{
  "code_works": 0.0-1.0,
  "meets_requirements": 0.0-1.0,
  "creative_output": 0.0-1.0,
  "explanation": "Brief explanation of scores"
}}"""

        judge_response = self._call_llm(self.judge_model, judge_prompt, max_tokens=512)

        try:
            json_start = judge_response.find('{')
            json_end = judge_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                scores = json.loads(judge_response[json_start:json_end])
                return scores
        except json.JSONDecodeError:
            pass

        # Default scores based on code execution
        return {
            "code_works": 0.7 if code_result['runs'] else 0.2,
            "meets_requirements": 0.5,
            "creative_output": 0.5,
            "explanation": "Could not parse judge response"
        }

    def evaluate_model(self, model: str, limit: int = None) -> dict:
        """Evaluate a model on all creative+technical prompts."""
        results = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "challenges": [],
            "overall_score": 0.0
        }

        all_scores = []
        prompts_to_eval = self.prompts["prompts"][:limit] if limit else self.prompts["prompts"]

        for prompt_data in prompts_to_eval:
            print(f"  Evaluating: {prompt_data['id']}")

            # Get model response
            response = self._call_llm(model, prompt_data["prompt"])

            # Determine language from prompt
            language = "javascript" if "JavaScript" in prompt_data["prompt"] else "python"

            # Extract and test code
            code = self._extract_code(response, language)
            code_result = self._test_code_execution(code, language)

            # Judge creative aspects
            scores = self._judge_creative(prompt_data, response, code_result)

            # Calculate weighted score
            weighted_score = (
                scores.get("code_works", 0) * 0.4 +
                scores.get("meets_requirements", 0) * 0.3 +
                scores.get("creative_output", 0) * 0.3
            )

            results["challenges"].append({
                "prompt_id": prompt_data["id"],
                "difficulty": prompt_data["difficulty"],
                "response_preview": response[:500] + "..." if len(response) > 500 else response,
                "code_extracted": bool(code),
                "code_runs": code_result["runs"],
                "scores": scores,
                "weighted_score": round(weighted_score, 3)
            })

            all_scores.append(weighted_score)

        if all_scores:
            results["overall_score"] = round(sum(all_scores) / len(all_scores), 3)

        return results

    def save_results(self, results: dict, output_dir: str = None) -> str:
        """Save evaluation results to JSON file."""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "results" / datetime.now().strftime("%Y-%m-%d")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = results["model"].replace("/", "-").replace(":", "-")
        output_path = output_dir / f"creative-technical-{model_name}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return str(output_path)


def main():
    """Run the creative+technical benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM on creative+technical challenges")
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Model to evaluate")
    parser.add_argument("--limit", type=int, help="Limit number of prompts")
    parser.add_argument("--output", help="Output directory for results")

    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Evaluating model: {args.model}")
    print("-" * 50)

    evaluator = CreativeTechnicalEvaluator()
    results = evaluator.evaluate_model(args.model, limit=args.limit)

    output_path = evaluator.save_results(results, args.output)

    print("-" * 50)
    print(f"Overall Score: {results['overall_score']}")
    print(f"Results saved to: {output_path}")

    print("\nChallenge Scores:")
    for challenge in results["challenges"]:
        status = "✓" if challenge["code_runs"] else "✗"
        print(f"  {status} {challenge['prompt_id']}: {challenge['weighted_score']}")


if __name__ == "__main__":
    main()
