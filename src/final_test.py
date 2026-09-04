import pandas as pd
from pathlib import Path

from predict import predict_text


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

TEST_CASES = [
    {
        "id": 1,
        "text": (
            "Scientists announced a new medical breakthrough "
            "after a five-year international study involving "
            "hospitals across Europe."
        ),
        "description": "Normal factual-style news text"
    },
    {
        "id": 2,
        "text": (
            "BREAKING: The government is secretly controlling "
            "the weather using hidden satellites!"
        ),
        "description": "Sensational misleading-style text"
    },
    {
        "id": 3,
        "text": (
            "Local man becomes billionaire after discovering "
            "that sleeping 14 hours a day increases productivity."
        ),
        "description": "Satirical-style text"
    },
    {
        "id": 4,
        "text": (
            "This sponsored article presents the newest smartphone "
            "as the greatest technological innovation of the year."
        ),
        "description": "Promotional-style text"
    },
    {
        "id": 5,
        "text": (
            "The city council approved the new transportation plan "
            "after a public meeting on Monday."
        ),
        "description": "Neutral news-style text"
    },
    {
        "id": 6,
        "text": (
            "Officials manipulated the image before publishing it "
            "to make the crowd appear significantly larger."
        ),
        "description": "Manipulation-related text"
    }
]


print("=" * 75)
print("FINAL SYSTEM TEST")
print("=" * 75)

rows = []

for case in TEST_CASES:

    print(f"\nTest Case {case['id']}")
    print("-" * 50)

    print("Description:")
    print(case["description"])

    print("\nInput:")
    print(case["text"])

    try:

        result = predict_text(
            case["text"]
        )

        predicted_label = result[
            "predicted_label"
        ]

        confidence = (
            result["confidence"] * 100
        )

        sentiment = result[
            "sentiment"
        ]["label"]

        compound = result[
            "sentiment"
        ]["compound"]

        print("\nPrediction:")
        print(predicted_label)

        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        print(
            "Sentiment:",
            sentiment
        )

        print(
            "Compound score:",
            f"{compound:.4f}"
        )

        rows.append(
            {
                "Test_ID": case["id"],
                "Description": (
                    case["description"]
                ),
                "Input_Text": (
                    case["text"]
                ),
                "Predicted_Label": (
                    predicted_label
                ),
                "Confidence_Percent": (
                    round(
                        confidence,
                        2
                    )
                ),
                "Sentiment": (
                    sentiment
                ),
                "Sentiment_Compound": (
                    compound
                ),
                "Status": "PASS"
            }
        )

    except Exception as error:

        print(
            "\nERROR:",
            error
        )

        rows.append(
            {
                "Test_ID": case["id"],
                "Description": (
                    case["description"]
                ),
                "Input_Text": (
                    case["text"]
                ),
                "Predicted_Label": "",
                "Confidence_Percent": "",
                "Sentiment": "",
                "Sentiment_Compound": "",
                "Status": "FAIL"
            }
        )


results_df = pd.DataFrame(
    rows
)

output_file = (
    RESULTS_DIR /
    "final_system_test.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 75)
print("FINAL TEST SUMMARY")
print("=" * 75)

print(
    results_df[
        [
            "Test_ID",
            "Predicted_Label",
            "Confidence_Percent",
            "Sentiment",
            "Status"
        ]
    ].to_string(
        index=False
    )
)

passed = (
    results_df[
        "Status"
    ] == "PASS"
).sum()

total = len(
    results_df
)

print(
    f"\nPassed: {passed}/{total}"
)

print(
    "\nSaved:",
    output_file
)

print("\n" + "=" * 75)
print("FINAL SYSTEM TEST COMPLETED")
print("=" * 75)