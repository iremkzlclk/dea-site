# -*- coding: utf-8 -*-
"""
DEA MODULU (CCR + BCC, iki asamali, girdi-yonelimli)
======================================================
GAMS kodu (dea_ccr_bcc_2024_fixed3.gms) ile BIREBIR AYNI formulasyon.
TEK BIR DONEM icin calisir - pipeline bunu her donem icin ayri cagirir.

Asama 1: theta(k) minimize edilir (radyal etkinlik)
   CCR: sum(j, lambda(j,k)*x(j,r)) <= theta(k)*x(k,r)   (girdi kisiti)
        sum(j, lambda(j,k)*y(j,s)) >= y(k,s)             (cikti kisiti)
   BCC: CCR + sum(j, lambda(j,k)) = 1  (konvekslik kisiti)

Asama 2: theta sabitken slack'ler (sx, sy) maksimize edilir
"""
import pulp
import pandas as pd


def solve_dea_period(x_df: pd.DataFrame, y_df: pd.DataFrame):
    """
    x_df: index=DMU, columns=girdi isimleri
    y_df: index=DMU, columns=cikti isimleri
    (x_df ve y_df ayni DMU sirasina/setine sahip olmali)

    Returns: dict with theta_ccr, theta_bcc, olcek_etkinligi (Series, index=DMU),
             lambda_ccr, lambda_bcc (DataFrame, index=peer j, columns=k),
             slack_x_ccr, slack_y_ccr, slack_x_bcc, slack_y_bcc (DataFrame, index=DMU)
    """
    dmus = list(x_df.index)
    girdiler = list(x_df.columns)
    ciktilar = list(y_df.columns)

    def stage1(vrs: bool):
        theta = {}
        lam = {}
        for k in dmus:
            prob = pulp.LpProblem(f"stage1_{k}", pulp.LpMinimize)
            th = pulp.LpVariable("theta", lowBound=0)
            l = {j: pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in dmus}
            prob += th
            for r in girdiler:
                prob += pulp.lpSum(l[j] * x_df.loc[j, r] for j in dmus) <= th * x_df.loc[k, r]
            for s in ciktilar:
                prob += pulp.lpSum(l[j] * y_df.loc[j, s] for j in dmus) >= y_df.loc[k, s]
            if vrs:
                prob += pulp.lpSum(l[j] for j in dmus) == 1
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            theta[k] = pulp.value(th)
            lam[k] = {j: pulp.value(l[j]) for j in dmus}
        return pd.Series(theta, name="theta"), pd.DataFrame(lam)  # columns=k, index=j

    def stage2(theta_fix: pd.Series, vrs: bool):
        sx_all, sy_all, lam_all = {}, {}, {}
        for k in dmus:
            prob = pulp.LpProblem(f"stage2_{k}", pulp.LpMaximize)
            l = {j: pulp.LpVariable(f"lambda_{j}", lowBound=0) for j in dmus}
            sx = {r: pulp.LpVariable(f"sx_{r}", lowBound=0) for r in girdiler}
            sy = {s: pulp.LpVariable(f"sy_{s}", lowBound=0) for s in ciktilar}
            prob += pulp.lpSum(sx.values()) + pulp.lpSum(sy.values())
            for r in girdiler:
                prob += pulp.lpSum(l[j] * x_df.loc[j, r] for j in dmus) + sx[r] == theta_fix[k] * x_df.loc[k, r]
            for s in ciktilar:
                prob += pulp.lpSum(l[j] * y_df.loc[j, s] for j in dmus) - sy[s] == y_df.loc[k, s]
            if vrs:
                prob += pulp.lpSum(l[j] for j in dmus) == 1
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            sx_all[k] = {r: pulp.value(sx[r]) for r in girdiler}
            sy_all[k] = {s: pulp.value(sy[s]) for s in ciktilar}
            lam_all[k] = {j: pulp.value(l[j]) for j in dmus}
        return (pd.DataFrame(sx_all).T, pd.DataFrame(sy_all).T, pd.DataFrame(lam_all))

    theta_ccr, _ = stage1(vrs=False)
    theta_bcc, _ = stage1(vrs=True)
    olcek_etkinligi = theta_ccr / theta_bcc

    slack_x_ccr, slack_y_ccr, lambda_ccr = stage2(theta_ccr, vrs=False)
    slack_x_bcc, slack_y_bcc, lambda_bcc = stage2(theta_bcc, vrs=True)

    return {
        "theta_ccr": theta_ccr,
        "theta_bcc": theta_bcc,
        "olcek_etkinligi": olcek_etkinligi,
        "lambda_ccr": lambda_ccr,
        "lambda_bcc": lambda_bcc,
        "slack_x_ccr": slack_x_ccr,
        "slack_y_ccr": slack_y_ccr,
        "slack_x_bcc": slack_x_bcc,
        "slack_y_bcc": slack_y_bcc,
    }


if __name__ == "__main__":
    # GAMS kodundaki 2024 - 12 DMU verisiyle test
    x = pd.DataFrame({
        "SimSuresi": [382, 334, 380, 292, 413, 363, 273, 348, 228, 322, 234, 271],
        "MuhSaat":   [364, 383, 416, 349, 351, 337, 420, 278, 370, 340, 377, 491],
        "Maliyet":   [316, 274, 311, 234, 317, 299, 300, 223, 334, 303, 272, 391],
    }, index=["A1","A2","B1","B2","F1","F2","C1","C2","D1","D2","E1","E2"])

    y = pd.DataFrame({
        "Hata":      [14, 10, 13, 15, 11, 11, 8, 7, 10, 8, 11, 9],
        "Dogruluk":  [92, 88, 89, 88, 85, 88, 87, 79, 91, 91, 86, 85],
        "AzaltProt": [4, 3, 3, 4, 4, 4, 3, 2, 3, 3, 2, 3],
    }, index=x.index)

    sonuc = solve_dea_period(x, y)
    print("theta_ccr:\n", sonuc["theta_ccr"].round(4))
    print("\ntheta_bcc:\n", sonuc["theta_bcc"].round(4))
    print("\nolcek_etkinligi:\n", sonuc["olcek_etkinligi"].round(4))
