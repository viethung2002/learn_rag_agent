import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze offline RAG evaluation results.")
    parser.add_argument(
        "--input",
        default=r"D:\japan_learn\full-stack-fastapi-template\arxiv-paper-curator\tests\data\evaluation_results.json",
        help="Input JSONL results path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: File not found at {input_path}")
        return

    llm_judge_scores = []
    ragas_metrics_scores = defaultdict(list)
    total_rows = 0
    missing_eval_rows = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON on line {line_num}")
                continue

            total_rows += 1
            evaluation = row.get("evaluation")

            if not evaluation:
                missing_eval_rows += 1
                continue

            # LLM-as-judge Score
            llm_judge = evaluation.get("llm_judge")
            if llm_judge and "score" in llm_judge:
                llm_judge_scores.append(llm_judge["score"])

            # RAGAS Metrics
            ragas_metrics = evaluation.get("ragas_metrics", {})
            for metric_name, value in ragas_metrics.items():
                if isinstance(value, (int, float)):
                    ragas_metrics_scores[metric_name].append(value)

    print("=" * 50)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total rows analyzed: {total_rows}")
    if missing_eval_rows > 0:
        print(f"Rows missing evaluation data: {missing_eval_rows}")
    print("-" * 50)

    # Calculate and print LLM Judge summary
    if llm_judge_scores:
        avg_llm_score = statistics.mean(llm_judge_scores)
        print(f"LLM-as-judge Average Score : {avg_llm_score:.2%} ({avg_llm_score:.4f})")
    else:
        print("LLM-as-judge Average Score : N/A (No data found)")

    print("-" * 50)

    # Calculate and print RAGAS summary
    if ragas_metrics_scores:
        print("RAGAS Metrics Average:")
        for metric_name, scores in ragas_metrics_scores.items():
            avg_metric_score = statistics.mean(scores)
            print(f"  - {metric_name:<20}: {avg_metric_score:.2%} ({avg_metric_score:.4f})")
    else:
        print("RAGAS Metrics Average      : N/A (No data found)")
    print("=" * 50)


if __name__ == "__main__":
    main()
