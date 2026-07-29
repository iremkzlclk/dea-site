# -*- coding: utf-8 -*-
"""
MALMQUIST MODULU (gecikmeli, girdi-yonelimli mesafe fonksiyonu)
==================================================================
GAMS kodu (malmquist_gecikmeli.gms) ile BIREBIR AYNI formulasyon,
COKLU girdi/cikti icin genellestirildi (orijinal GAMS ornegi tek
girdi/cikti kullaniyordu, ama LP yapisi zaten genel).

D(k,t,tp): k DMU'sunun t donemindeki (x,y) verisinin, tp donemi
frontier'ina gore mesafe fonksiyonu.

EC(k,t) = D(k,tp,tp) / D(k,t,t)
TC(k,t) = sqrt( [D(k,t,t)/D(k,t,tp)] * [D(k,tp,t)/D(k,tp,tp)] )
M(k,t)  = EC(k,t) * TC(k,t)          -- t = GECISIN BASLADIGI donem
"""
import pulp
import pandas as pd
import itertools


def _distance(k, t, tp, X: dict, Y: dict, dmus, girdiler, ciktilar):
    """D(k,t,tp): k'nin t donemi verisi, tp donemi frontier'ina gore."""
    prob = pulp.LpProblem("dist", pulp.LpMinimize)
    theta = pulp.LpVariable("theta", lowBound=0)
    lam = {i: pulp.LpVariable(f"lambda_{i}", lowBound=0) for i in dmus}
    prob += theta
    for r in girdiler:
        prob += pulp.lpSum(lam[i] * X[tp].loc[i, r] for i in dmus) <= theta * X[t].loc[k, r]
    for s in ciktilar:
        prob += pulp.lpSum(lam[i] * Y[tp].loc[i, s] for i in dmus) >= Y[t].loc[k, s]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return pulp.value(theta)


def solve_malmquist(X: dict, Y: dict, periods: list):
    """
    X, Y: dict {donem_adi: DataFrame(index=DMU, columns=girdi/cikti)}
    periods: donemlerin SIRALI listesi, orn. ["t1","t2","t3","t4"]

    Returns: DataFrame(index=[DMU, donem], columns=[EC, TC, M])
             -- sadece ARDISIK gecisler icin (t_i -> t_{i+1})
    """
    dmus = list(X[periods[0]].index)
    girdiler = list(X[periods[0]].columns)
    ciktilar = list(Y[periods[0]].columns)

    # ihtiyac duyulan (k,t,tp) uclulerini onceden belirle: sadece t==tp
    # ve ardisik gecislerdeki (t,t),(tp,tp),(t,tp),(tp,t) kombinasyonlari
    trans = list(zip(periods[:-1], periods[1:]))
    needed = set()
    for t, tp in trans:
        needed.update([(t, t), (tp, tp), (t, tp), (tp, t)])

    dist = {}
    for k in dmus:
        for (t, tp) in needed:
            dist[(k, t, tp)] = _distance(k, t, tp, X, Y, dmus, girdiler, ciktilar)

    rows = []
    for t, tp in trans:
        for k in dmus:
            d_tt = dist[(k, t, t)]
            d_tptp = dist[(k, tp, tp)]
            d_ttp = dist[(k, t, tp)]
            d_tpt = dist[(k, tp, t)]
            EC = d_tptp / d_tt
            TC = ((d_tt / d_ttp) * (d_tpt / d_tptp)) ** 0.5
            M = EC * TC
            rows.append({"DMU": k, "donem": t, "EC": EC, "TC": TC, "M": M})

    return pd.DataFrame(rows).set_index(["DMU", "donem"])


if __name__ == "__main__":
    # GAMS malmquist_gecikmeli.gms verisiyle (tek girdi x1_SimSuresi, tek cikti y2_Prototip) test
    dmus = ["Turbin_Kanadi","Yanma_Odasi","Kompresor_Diski","Egzoz_Nozulu","Disli_Kutusu",
            "Sogutma_Kanali","Mil","Rulman","Pervane","Fan_Kanadi","Yakit_Enjektoru",
            "Turbin_Diski","Kompresor_Kanadi","Egzoz_Konisi","Yaglama_Sistemi"]

    x1 = [420,500,300,260,340,220,380,480,150,430,200,350,280,240,160]
    x2 = [400,520,310,250,360,230,370,190,440,410,190,340,270,235,155]
    x3 = [410,490,290,270,330,210,390,170,460,400,180,330,260,225,150]
    x4 = [390,510,320,240,350,240,360,185,430,390,175,320,255,220,145]
    y1 = [3,3,2,2,2,1,2,1,3,2,1,2,2,2,1]
    y2 = [3,3,2,2,2,1,2,1,3,3,1,2,2,2,1]
    y3 = [4,3,2,2,2,1,3,1,3,3,2,3,2,2,1]
    y4 = [4,3,3,2,2,1,2,1,4,3,2,3,2,2,1]

    X = {f"t{i+1}": pd.DataFrame({"x1_SimSuresi": v}, index=dmus) for i, v in enumerate([x1,x2,x3,x4])}
    Y = {f"t{i+1}": pd.DataFrame({"y2_Prototip": v}, index=dmus) for i, v in enumerate([y1,y2,y3,y4])}

    sonuc = solve_malmquist(X, Y, ["t1","t2","t3","t4"])
    pd.set_option("display.width", 160)
    print(sonuc.round(4))
