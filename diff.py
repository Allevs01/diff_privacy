import opendp.prelude as dp
import pandas as pd
import urllib.request

# enable contrib functionality
dp.enable_features('contrib')

def main():
    privacy_unit = dp.unit_of(contributions=1)
    input_metric, d_in = privacy_unit
    #0.01,0.1, 0.5, 0.9
    privacy_loss = dp.loss_of(epsilon=1.)
    privacy_measure, d_out = privacy_loss

    col_names = [
        "name", "sex", "age", "maritalStatus", "hasChildren", "highestEducationLevel", 
        "sourceOfStress", "smoker", "optimism", "lifeSatisfaction", "selfEsteem"
    ]

    data_url = "https://raw.githubusercontent.com/opendp/opendp/sydney/teacher_survey.csv"
    with urllib.request.urlopen(data_url) as data_req:
        data = data_req.read().decode('utf-8')

    # Convert the data into a pandas DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(data), names=col_names, skiprows=1)

    # Print a preview of the data
    print("Data Preview:")
    print(df.head())

    # Calculate the count of the "age" column
    age_count = df["age"].count()

    # Print the simple count
    print(f"Simple Count of Age: {age_count}")

    context = dp.Context.compositor(
        data=data,
        privacy_unit=privacy_unit,
        privacy_loss=privacy_loss,
        split_evenly_over=3
    )

    count_query = (
        context.query()
        .split_dataframe(",", col_names=col_names)
        .select_column("age", str)  # temporary until OpenDP 0.10 (Polars dataframe)
        .count()
        .laplace()
    )

    scale = count_query.param()
    
    # Print the laplace noise scale 
    print(f"Laplace Noise Scale: {scale}")

    accuracy = dp.discrete_laplacian_scale_to_accuracy(scale=scale, alpha=0.05)

    # Print the accuracy interval
    print(f"Accuracy Interval: {accuracy}")

    dp_count = count_query.release()
    interval = (dp_count - accuracy, dp_count + accuracy)

    # Print the differentially private count and its accuracy interval
    print(f"Differentially Private Count: {dp_count}")
    print(f"Count Interval: {interval}")

    # Convert the "age" column to numeric values
    df["age"] = pd.to_numeric(df["age"], errors='coerce')

    # Calculate the mean of the "age" column, ignoring NaN values
    age_mean = df["age"].mean()

    # Print the simple mean
    print(f"Simple Mean of Age: {age_mean}")

    mean_query = (
        context.query()
        .split_dataframe(",", col_names=col_names)
        .select_column("age", str)
        .cast_default(float)
        .clamp((18.0, 70.0))  # a best-guess based on public information
        # Explanation for `constant=42`:
        #    since dp_count may be larger than the true size, 
        #    imputed rows will be given an age of 42.0 
        #    (also a best guess based on public information)
        .resize(size=dp_count, constant=42.0)
        .mean()
        .laplace()
    )

    dp_mean = mean_query.release()

    # Print the differentially private mean
    print(f"Differentially Private Mean Age: {dp_mean}")

if __name__ == "__main__":
    main()