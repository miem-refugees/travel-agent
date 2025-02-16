import pandas as pd


def processed_theatres(theatres: pd.DataFrame) -> pd.DataFrame:
    processed_theaters = theatres[["CommonName", "ObjectAddress", "geoData"]]
    processed_theaters["District"] = processed_theaters["ObjectAddress"].map(
        lambda x: x[0]["District"]
    )
    processed_theaters["Address"] = processed_theaters["ObjectAddress"].map(
        lambda x: x[0]["Address"].replace("Российская Федерация, город ", "")
    )
    processed_theaters.drop(["ObjectAddress"], axis=1, inplace=True)
    processed_theaters.rename(columns={"CommonName": "Name"}, inplace=True)

    return processed_theaters
