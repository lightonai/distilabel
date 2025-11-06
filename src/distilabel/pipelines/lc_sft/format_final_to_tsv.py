import pandas as pd
from distilabel import utils


# Absolute paths per workspace preference
OFFICIAL_TSV_PATH = \
    "/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/correct_mm_long_doc/MMLongBench_DOC.tsv"
FINAL_JSON_PATH = \
    "/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/correct_mm_long_doc/final.json"
OUTPUT_TSV_PATH = \
    "/mnt/nfs/austin_shared/mp_data_gen/distilabel/out/correct_mm_long_doc/MMLongBench_DOC_corrected.tsv"


def main() -> None:
    df = pd.read_csv(OFFICIAL_TSV_PATH, sep="\t")

    if "question" not in df.columns:
        raise KeyError("Expected 'question' column in official TSV")
    answer_col = "answer" if "answer" in df.columns else None
    if answer_col is None:
        raise KeyError("Expected 'answer' column in official TSV")

    final_rows = utils.load_json(FINAL_JSON_PATH)

    # Map original question -> (new_question, new_answer)
    og_question_to_updates = {
        row["og_question"]: (row["question"], row["answer"]) for row in final_rows
    }

    mask = df["question"].isin(og_question_to_updates)
    if mask.any():
        original_questions = df.loc[mask, "question"].copy()
        df.loc[mask, "question"] = original_questions.map(lambda q: og_question_to_updates[q][0])
        df.loc[mask, answer_col] = original_questions.map(lambda q: og_question_to_updates[q][1])

    df.to_csv(OUTPUT_TSV_PATH, sep="\t", index=False)


if __name__ == "__main__":
    main()


