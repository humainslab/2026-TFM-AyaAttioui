# CHANGES THAT I MADE (Aya Attioui):
# 1. Added compute_top_k() — dynamic top_k instead of hardcoded 3
# 2. Renamed top3_* functions to topk_*
# 3. Replaced [1,2,3] with list(range(1, k+1)) everywhere
# 4. Replaced /3 with /k everywhere
# 5. calculate_metrics_for_indices() now computes top_k automatically
# 6. Removed create_metrics_dfs() — not needed for this TFM


import pandas as pd
import re
import dalex as dx


def evaluate_value(value):
    if value >= 0:
        return "Positive"
    else:
        return "Negative"



# computes dynamic top_k
def compute_top_k(num_features):
    return max(2, min(10, round(num_features * 0.3)))


# CHANGE: parameter renamed from m to k
def preprocess_dfs(breakdown_df, shap_df, lime_df, n, k):

    columns_multi = pd.MultiIndex.from_tuples([
        ('Breakdown', 'Ranking'), ('Breakdown', 'Sign'),
        ('Shap', 'Ranking'), ('Shap', 'Sign'),
        ('Lime', 'Ranking'), ('Lime', 'Sign')])

    pattern = re.compile(r'(?:<=|<|>=|>)\s*(\w+)\s*(?:<=|<|>=|>)')

    
    breakdown_df = breakdown_df.loc[:, ['variable_name', 'contribution', 'sign']]
    breakdown_df = breakdown_df.drop(index=[0, n+1])
    breakdown_df['sign'] = breakdown_df['sign'].replace({1.0: 'Positive', 0.0: 'Null', -1.0: 'Negative'})
    breakdown_df = breakdown_df.sort_values(by='contribution', key=lambda x: abs(x), ascending=False)

    
    shap_df = shap_df.loc[:, ['variable_name', 'contribution', 'sign']]
    shap_df = shap_df.tail(n)
    shap_df['sign'] = shap_df['sign'].replace({1.0: 'Positive', 0.0: 'Null', -1.0: 'Negative'})
    shap_df = shap_df.sort_values(by='contribution', key=lambda x: abs(x), ascending=False)

    
    lime_list = []
    for feature in lime_df['variable']:
        if pattern.findall(feature):
            match = pattern.search(feature)
            if match:
                lime_list.append(match.group(1))
        else:
            splits = feature.split(" ")
            lime_list.append(splits[0])

    lime_df["Feature"] = lime_list
    lime_df["Sign"] = lime_df["effect"].apply(evaluate_value)
    lime_df = lime_df.sort_values(by='effect', key=lambda x: abs(x), ascending=False)
    lime_df = lime_df.drop(columns=['variable'])

    
    breakdown_df['Ranking'] = breakdown_df['contribution'].abs().rank(ascending=False).astype(int)
    breakdown_df.rename(columns={'sign': 'Sign', 'variable_name': 'Feature'}, inplace=True)
    breakdown_df = breakdown_df[['Feature', 'Ranking', 'contribution', 'Sign']]

    shap_df['Ranking'] = shap_df['contribution'].abs().rank(ascending=False).astype(int)
    shap_df.rename(columns={'sign': 'Sign', 'variable_name': 'Feature'}, inplace=True)
    shap_df = shap_df[['Feature', 'Ranking', 'contribution', 'Sign']]

    lime_df['Ranking'] = lime_df['effect'].abs().rank(ascending=False).astype(int)
    lime_df = lime_df.head(k)  # CHANGE: was head(m)
    lime_df = lime_df[['Feature', 'Ranking', 'effect', 'Sign']]

    breakdown_df = breakdown_df.drop(columns=['contribution'])
    shap_df = shap_df.drop(columns=['contribution'])
    lime_df = lime_df.drop(columns=['effect'])

    breakdown_df = breakdown_df.head(k).reset_index(drop=True)  
    shap_df = shap_df.head(k).reset_index(drop=True)            
    lime_df = lime_df.reset_index(drop=True)

    
    breakdown_features = list(breakdown_df['Feature'])
    shap_features = list(shap_df['Feature'])
    lime_features = list(lime_df['Feature'])
    all_features = list(set(breakdown_features + shap_features + lime_features))
    all_features = all_features[::-1]

    df_final = pd.DataFrame(index=range(len(all_features)), columns=columns_multi)
    df_final['Feature'] = list(all_features)
    df_final = df_final[['Feature', 'Breakdown', 'Shap', 'Lime']]

    for feature in df_final['Feature']:

        breakdown_row = breakdown_df[breakdown_df['Feature'] == feature]
        if not breakdown_row.empty:
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Ranking')] = breakdown_row.iloc[0]['Ranking']
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Sign')]    = breakdown_row.iloc[0]['Sign']
        else:
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Ranking')] = '-'
            df_final.loc[df_final['Feature'] == feature, ('Breakdown', 'Sign')]    = '-'

        shap_row = shap_df[shap_df['Feature'] == feature]
        if not shap_row.empty:
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Ranking')] = shap_row.iloc[0]['Ranking']
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Sign')]    = shap_row.iloc[0]['Sign']
        else:
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Ranking')] = '-'
            df_final.loc[df_final['Feature'] == feature, ('Shap', 'Sign')]    = '-'

        lime_row = lime_df[lime_df['Feature'] == feature]
        if not lime_row.empty:
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Ranking')] = lime_row.iloc[0]['Ranking']
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Sign')]    = lime_row.iloc[0]['Sign']
        else:
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Ranking')] = '-'
            df_final.loc[df_final['Feature'] == feature, ('Lime', 'Sign')]    = '-'

    ranking_breakdown = df_final[('Breakdown', 'Ranking')]
    values_valid = ranking_breakdown[ranking_breakdown != '-']
    cases_invalid = ranking_breakdown[ranking_breakdown == '-'].index
    cases_valid = []
    cases_invalid_restantes = list(cases_invalid)
    cases_valid.extend(values_valid.sort_values().index.tolist())

    for cas in cases_invalid:
        value_shap = df_final.at[cas, ('Shap', 'Ranking')]
        if value_shap != '-':
            cases_valid.extend([cas])
            cases_invalid_restantes.remove(cas)

    for cas in cases_invalid_restantes:
        value_lime = df_final.at[cas, ('Lime', 'Ranking')]
        if value_lime != '-':
            cases_valid.extend([cas])

    df_final = df_final.loc[cases_valid]
    df_final.set_index('Feature', inplace=True)

    return breakdown_df, shap_df, lime_df, df_final



def topk_features(df, case, k):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_topk = set(df[df[(tech, 'Ranking')].isin(list(range(1, k+1)))].index)  # CHANGE
        set_expl[tech] = inst_topk

    result_case = {
        f'TOP{k} {tech}': ', '.join(set_expl[tech]) for tech in ['Breakdown', 'Shap', 'Lime']  # CHANGE
    }

    for tech1, inst_topk_1 in set_expl.items():
        for tech2, inst_topk_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'
                result_case[col_name] = len(inst_topk_1.intersection(inst_topk_2)) / k  # CHANGE: /3 → /k

    df_topk_case = pd.DataFrame(result_case, index=[case])
    return df_topk_case



def topk_rank(df, case, k):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_topk = df[df[(tech, 'Ranking')].isin(list(range(1, k+1)))]  # CHANGE
        inst_topk_str = ', '.join([f'{ranking}:{inst}' for ranking, inst in zip(inst_topk[(tech, 'Ranking')], inst_topk.index)])
        set_expl[tech] = inst_topk_str

    result_case = {
        f'TOP{k} {tech}': set_expl[tech] for tech in ['Breakdown', 'Shap', 'Lime']  # CHANGE
    }

    for tech1, inst_topk_1 in set_expl.items():
        for tech2, inst_topk_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'
                coincidence = [inst for inst in inst_topk_1.split(', ') if inst in inst_topk_2.split(', ')]
                result_case[col_name] = len(coincidence) / k  # CHANGE: /3 → /k

    df_topk_rank = pd.DataFrame(result_case, index=[case])
    return df_topk_rank



def topk_sign(df, case, k):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_topk = df[df[(tech, 'Ranking')].isin(list(range(1, k+1)))]  # CHANGE
        inst_topk_str = ', '.join([f'{inst}:{sign}' for inst, sign in zip(inst_topk.index, inst_topk[(tech, 'Sign')])])
        set_expl[tech] = inst_topk_str

    result_case = {
        f'TOP{k} {tech}': set_expl[tech] for tech in ['Breakdown', 'Shap', 'Lime']  # CHANGE
    }

    for tech1, inst_topk_1 in set_expl.items():
        for tech2, inst_topk_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'
                coincidence = [inst for inst in inst_topk_1.split(', ') if inst in inst_topk_2.split(', ')]
                result_case[col_name] = len(coincidence) / k  # CHANGE: /3 → /k

    df_topk_sign = pd.DataFrame(result_case, index=[case])
    return df_topk_sign



def topk_rank_sign(df, case, k):
    set_expl = {}

    for tech in ['Breakdown', 'Shap', 'Lime']:
        inst_topk = df[df[(tech, 'Ranking')].isin(list(range(1, k+1)))]  # CHANGE
        inst_topk_str = ', '.join([f'{ranking}:{inst}:{sign}' for ranking, inst, sign in zip(inst_topk[(tech, 'Ranking')], inst_topk.index, inst_topk[(tech, 'Sign')])])
        set_expl[tech] = inst_topk_str

    result_case = {
        f'TOP{k} {tech}': set_expl[tech] for tech in ['Breakdown', 'Shap', 'Lime']  # CHANGE
    }

    for tech1, inst_topk_1 in set_expl.items():
        for tech2, inst_topk_2 in set_expl.items():
            if tech1 < tech2:
                col_name = f'{tech1}-{tech2}'
                coincidence = [inst for inst in inst_topk_1.split(', ') if inst in inst_topk_2.split(', ')]
                result_case[col_name] = len(coincidence) / k  # CHANGE: /3 → /k

    df_topk_rank_sign = pd.DataFrame(result_case, index=[case])
    return df_topk_rank_sign



def calculate_metrics(df_list):
    df_final = pd.concat(df_list)

    mean_bd_shap   = df_final['Breakdown-Shap'].mean()
    mean_bd_lime   = df_final['Breakdown-Lime'].mean()
    mean_lime_shap = df_final['Lime-Shap'].mean()

    return df_final, mean_bd_shap, mean_bd_lime, mean_lime_shap



def calculate_metrics_for_indices(models_dict, indexes_dict, X_test, X_train, y_train, num_features):

    # CHANGE: dynamic top_k instead of fixed 3
    top_k = compute_top_k(num_features)
    print(f"  top_k = {top_k} (based on {num_features} features)")

    results_dict = {}
    metrics_dict = {}

    for model_name, model in models_dict.items():
        results_dict[model_name] = {}
        metrics_dict[model_name] = {}

        exp = dx.Explainer(model, X_train, y_train, verbose=False)

        for index_name, indices in indexes_dict.items():
            results_dict[model_name][index_name] = {}
            metrics_dict[model_name][index_name] = {}
            model_results = results_dict[model_name][index_name]

            df_list_top       = []
            df_list_rank      = []
            df_list_sign      = []
            df_list_rank_sign = []

            for i in indices:
                instance = X_test.loc[i]

                # UNCHANGED
                breakdown = exp.predict_parts(instance, type="break_down", random_state=42)
                shap      = exp.predict_parts(instance, type="shap",       random_state=42)
                lime      = exp.predict_surrogate(instance,                random_state=42)

                breakdown_df = breakdown.result
                shap_df      = shap.result
                lime_df      = lime.result

                # CHANGE: top_k instead of top_num_features
                breakdown_df, shap_df, lime_df, df_final = preprocess_dfs(
                    breakdown_df, shap_df, lime_df, num_features, top_k
                )

                model_results[f"breakdown_df_{i}"] = breakdown_df
                model_results[f"shap_df_{i}"]      = shap_df
                model_results[f"lime_df_{i}"]      = lime_df
                model_results[f"df_final_{i}"]     = df_final

                
                metrics_dict[model_name][index_name][f"df_top_metric_{i}"]       = topk_features(df_final, i, top_k)
                metrics_dict[model_name][index_name][f'df_rank_metric_{i}']      = topk_rank(df_final, i, top_k)
                metrics_dict[model_name][index_name][f'df_sign_metric_{i}']      = topk_sign(df_final, i, top_k)
                metrics_dict[model_name][index_name][f'df_rank_sign_metric_{i}'] = topk_rank_sign(df_final, i, top_k)

                df_list_top.append(metrics_dict[model_name][index_name][f'df_top_metric_{i}'])
                df_list_rank.append(metrics_dict[model_name][index_name][f'df_rank_metric_{i}'])
                df_list_sign.append(metrics_dict[model_name][index_name][f'df_sign_metric_{i}'])
                df_list_rank_sign.append(metrics_dict[model_name][index_name][f'df_rank_sign_metric_{i}'])

            df_top_final,  mean_top_bd_shap,      mean_top_bd_lime,       mean_top_lime_shap       = calculate_metrics(df_list_top)
            df_rank_final, mean_rank_bd_shap,      mean_rank_bd_lime,      mean_rank_lime_shap      = calculate_metrics(df_list_rank)
            df_sign_final, mean_sign_bd_shap,      mean_sign_bd_lime,      mean_sign_lime_shap      = calculate_metrics(df_list_sign)
            df_rs_final,   mean_rank_sign_bd_shap, mean_rank_sign_bd_lime, mean_rank_sign_lime_shap = calculate_metrics(df_list_rank_sign)

            
            metrics_dict[model_name][index_name]['Mean_topk_BD_Shap']        = mean_top_bd_shap
            metrics_dict[model_name][index_name]['Mean_topk_BD_Lime']        = mean_top_bd_lime
            metrics_dict[model_name][index_name]['Mean_topk_Lime_Shap']      = mean_top_lime_shap
            metrics_dict[model_name][index_name]['df_top_metric_final']      = df_top_final

            metrics_dict[model_name][index_name]['Mean_rank_BD_Shap']        = mean_rank_bd_shap
            metrics_dict[model_name][index_name]['Mean_rank_BD_Lime']        = mean_rank_bd_lime
            metrics_dict[model_name][index_name]['Mean_rank_Lime_Shap']      = mean_rank_lime_shap
            metrics_dict[model_name][index_name]['df_rank_metric_final']     = df_rank_final

            metrics_dict[model_name][index_name]['Mean_sign_BD_Shap']        = mean_sign_bd_shap
            metrics_dict[model_name][index_name]['Mean_sign_BD_Lime']        = mean_sign_bd_lime
            metrics_dict[model_name][index_name]['Mean_sign_Lime_Shap']      = mean_sign_lime_shap
            metrics_dict[model_name][index_name]['df_sign_metric_final']     = df_sign_final

            metrics_dict[model_name][index_name]['Mean_rank_sign_BD_Shap']   = mean_rank_sign_bd_shap
            metrics_dict[model_name][index_name]['Mean_rank_sign_BD_Lime']   = mean_rank_sign_bd_lime
            metrics_dict[model_name][index_name]['Mean_rank_sign_Lime_Shap'] = mean_rank_sign_lime_shap
            metrics_dict[model_name][index_name]['df_rank_sign_metric_final']= df_rs_final

    return metrics_dict, results_dict