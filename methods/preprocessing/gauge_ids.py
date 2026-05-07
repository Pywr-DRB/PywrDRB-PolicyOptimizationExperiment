# Define gauges and mappings
from methods.config import RAW_DATA_DIR

inflow_gauges = {
    "prompton": ["01428750"],
    "beltzvilleCombined": ["01449360"],
    "fewalter": ["01447720", "01447500"],
}

storage_gauges = {
    "beltzvilleCombined": ["01449790"],
    "fewalter": ["01447780"],
    "prompton": ["01428900"],
    "blueMarsh": ["01470870"],
    # fill in others as needed
}

release_gauges = {
    "prompton": ["01429000"],
    "beltzvilleCombined": ["01449800"],
    "fewalter": ["01447800"],
    "blueMarsh": ["01470960"],
}

storage_curves = {
    "01449790": f"{RAW_DATA_DIR}/beltzvilleCombined_storage_curve.csv",  # "beltzvilleCombined"
    "01447780": f"{RAW_DATA_DIR}/fewalter_storage_curve.csv",  # fewalter
    "01428900": f"{RAW_DATA_DIR}/prompton_storage_curve.csv",  # prompton
    "01470870": f"{RAW_DATA_DIR}/blueMarsh_storage_curve.csv",  # blueMarsh
}
