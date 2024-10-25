import csv
import os


def client_write_results(
    filename,
    client_id,
    acc_local_model,
    f1_local_model,
    acc_global_model,
    f1_global_model,
    tam_test_data,
):
    """Function to write the results onto a CSV file.

    Args:
        filename (str): Filename to write the results
        client_id (str): Client ID generated in each execution
        acc_local_model (float): Accuracy of the local model on the local test data
        f1_local_model (float): Macro F1 of the local model on the local test data
        acc_local_model (float): Accuracy of the global model on the local test data
        f1_local_model (float): Macro F1 of the global model on the local test data
    """
    if not os.path.exists(filename):
        header = [
            "client_id",
            "local_model_acc",
            "local_model_f1",
            "global_model_acc",
            "global_model_f1",
            "tam_test_data",
        ]
        with open(filename, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(header)
    results = [
        client_id,
        acc_local_model,
        f1_local_model,
        acc_global_model,
        f1_global_model,
        tam_test_data,
    ]
    with open(filename, "a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(results)


def print_results_client(
    tam_test_data,
    acc_global_model,
    acc_local_model,
    f1_global_model,
    f1_local_model,
    report=None,
):
    """Function to print the results of a client

    Args:
        tam_test_data (int): Length of the local test data
        acc_global_model (float): Accuracy of the local model on the client test data
        acc_local_model (float): Accuracy of the global model on the client test data
        f1_global_model (float): Macro F1 of the local model on the client test data
        acc_global_model (float): Macro F1 of the global model on the client test data
        report (sklearn.classification_report, Optional): Classification report.
    """
    print("Results on clients test data that improves local results.")
    print(f"\tLen of test data on client: {tam_test_data}")
    print(f"\tAccuracy global model on test data: {acc_global_model}")
    print(f"\tAccuracy with local model: {acc_local_model}")
    print(f"\tMacro F1 global model on test data: {f1_global_model}")
    print(f"\tF1 macro with local model: {f1_local_model}")
    if report:
        print(f"Classification report: {report}")
