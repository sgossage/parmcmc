import numpy as np


def match_to_mist(filters):

    MIST_colors_prefixes = []
    for i, filtern in enumerate(filters):

        # HST_WFC3 uses the format e.g. WFC3_UVIS_F814W in MIST
        # This would be UVIS814W in MATCH. So...
        if "UVIS" in filtern:
            filters[i] = filtern.replace("UVIS", "WFC3_UVIS_F")
            MIST_colors_prefixes.append("HST_WFC3")
        if "F" in filtern:
            filters[i] = "ACS_WFC_"+filtern
            MIST_colors_prefixes.append("HST_ACSWF")

        # UBVRIplus uses Tycho_B, etc. -- the same 
        # as MATCH and UBVRIplus is the default 
        # prefix in MIST_codes, so all good.
        # (at least for Tycho filters...)
        elif "Tycho_B" in filtern:
            MIST_colors_prefixes.append("UBVRIplus")

        elif "Gaia_BP" in filtern:
            MIST_colors_prefixes.append("UBVRIplus")

    return filters, MIST_colors_prefixes
