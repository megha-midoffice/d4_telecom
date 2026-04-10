# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:15:12 2026

@author: MeghaGhosh_b5xx485
"""

def apply_dimension_scenario(
    forecast_df,
    dimension,
    pct_map
):
    """
    Apply multiple % adjustments within ONE dimension.

    Parameters
    ----------
    forecast_df : DataFrame
        Must contain:
        - predicted_amount
        - dimension column
    dimension : str
        e.g. 'CUSTOMER_TYPE'
    pct_map : dict
        Example:
        {
            'B2C': 5,
            'B2B': 15,
            'Unknown': -2
        }

    Returns
    -------
    dict with:
        baseline_total
        total_impact
        new_total
    """

    baseline_total = forecast_df["predicted_amount"].sum()

    total_impact = 0

    for category, pct_change in pct_map.items():

        category_total = forecast_df.loc[
            forecast_df[dimension] == category,
            "predicted_amount"
        ].sum()

        impact = category_total * (pct_change / 100)
        total_impact += impact

    new_total = baseline_total + total_impact

    return {
        "baseline_total": baseline_total,
        "total_impact": total_impact,
        "new_total": new_total
    }
