import pandas as pd


def processed_cultural(cultural: pd.DataFrame) -> pd.DataFrame:
    processed = cultural[
        ["EnsembleNameOnDoc", "District", "ObjectType", "Addresses", "geoData"]
    ]
    processed["EnsembleNameOnDoc"] = (
        processed["ObjectType"] + " " + processed["EnsembleNameOnDoc"]
    )
    processed.drop(["ObjectType"], axis=1, inplace=True)
    processed["Addresses"] = processed["Addresses"].map(lambda x: x.split(";")[0])
    processed["EnsembleNameOnDoc"] = processed["Addresses"].map(
        lambda x: x.split(";")[0]
    )
    processed.rename(
        columns={
            "EnsembleNameOnDoc": "Name",
            "Addresses": "Address",
        },
        inplace=True,
    )

    return processed
