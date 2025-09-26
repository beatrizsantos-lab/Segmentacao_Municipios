# -*- coding: utf-8 -*-

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Análise de Cluster - Segmentação Socioeconômica de Municípios para Alocação Eficiente de Recursos 
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# O script a seguir tem como objetivo realizar uma análise de cluster para segmentar os municípios brasileiros 
# com base em características socioeconômicas para TCC do MBA em Data Science e Analytics USP ESALQ.

# Autora: Beatriz Santos

# Data de Criação: 10/06/2025
# Última Modificação: 20/09/2025
# Versão: 2.0

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 1. INSTALAÇÃO E IMPORTS
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#%% >> Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin
!pip install openpyxl

#%% >> Importando os pacotes

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --> Pacotes para Clusterização e PCA
import scipy.cluster.hierarchy as sch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# --> Pacotes para Análise e Visualização
import pingouin as pg
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 2. CONFIGURAÇÃO E CONSOLIDAÇÃO DOS DADOS
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#%% >> Importando e Consolidando os Bancos de Dados

print("--- Consolidação de Dados ---")

caminho_pasta = r'C:\Users\beatr\OneDrive\Área de Trabalho\Workspace\Clustering'  #Caminho da pasta com os arquivos
#Planilhas a serem consolidadas:
arquivos_excel = [
    'Alfabetizacao2022.xlsx', 'Despesa_AssistSocial_Prev.xlsx', 'Despesa_Edu_Cult.xlsx',
    'Despesa_GestaoAmbAgric_Org.xlsx', 'Despesa_Sau_Sanea.xlsx', 'Despesa_SegPub_Prev.xlsx',
    'Despesa_Urban_Hab.xlsx', 'Ifdm.xlsx', 'Lixo2022.xlsx', 'PopulacaoPorCor2022.xlsx',
    'PopulacaoPorIdade2022.xlsx', 'PopulacaoTotal.xlsx', 'ReceitaCorrBruta.xlsx',
    'Saneamento2022.xlsx'
]
coluna_chave = 'Cod' #Identificador único dos municípios

df_consolidado = pd.read_excel(os.path.join(caminho_pasta, arquivos_excel[0]), dtype={coluna_chave: str})
for arquivo in arquivos_excel[1:]:
    df_temp = pd.read_excel(os.path.join(caminho_pasta, arquivo), dtype={coluna_chave: str})
    cols_para_remover = [col for col in df_temp.columns if col in df_consolidado.columns and col != coluna_chave]
    if cols_para_remover:
        df_temp.drop(columns=cols_para_remover, inplace=True)
    df_consolidado = pd.merge(df_consolidado, df_temp, on=coluna_chave, how='outer')

for col in df_consolidado.columns:
    if df_consolidado[col].dtype == 'object' and col != 'Municipio':
        df_consolidado[col] = pd.to_numeric(df_consolidado[col], errors='coerce')
df_consolidado.dropna(subset=[coluna_chave], inplace=True)
df_consolidado[coluna_chave] = df_consolidado[coluna_chave].astype(int)
print(f"Base consolidada com {df_consolidado.shape[0]} linhas e {df_consolidado.shape[1]} colunas.")

#%% 2.1. ENGENHARIA DE ATRIBUTOS

print("\n--- Criando Indicadores de Impacto ---")
df_indicadores = df_consolidado.copy()
try:
    # >> Indicadores Financeiros
    despesas_2022 = {'AS': 'DespAS_2022', 'Edu': 'DespEdu_2022', 'GA': 'DespGA_2022','SS': 'DespSS_2022', 'SP': 'DespSP_2022', 'UH': 'DespUH_2022'}
    for nome, coluna in despesas_2022.items():
        if coluna in df_indicadores.columns and 'Pop_Total' in df_indicadores.columns:
            df_indicadores[f'Desp_{nome}_PC'] = df_indicadores[coluna] / df_indicadores['Pop_Total']
        if coluna in df_indicadores.columns and 'Rec_2022' in df_indicadores.columns:
            df_indicadores[f'Prioridade_{nome}'] = df_indicadores[coluna] / df_indicadores['Rec_2022']
    # >> Indicadores Demográficos
    jovem_cols = ['Faixa_0_4', 'Faixa_5_9', 'Faixa_10_14']
    ativa_cols = ['Faixa_15_19', 'Faixa_20_24', 'Faixa_25_29', 'Faixa_30_34', 'Faixa_35_39', 'Faixa_40_44', 'Faixa_45_49', 'Faixa_50_54', 'Faixa_55_59', 'Faixa_60_64']
    idosa_cols = ['Faixa_65_69', 'Faixa_70_74', 'Faixa_75_79', 'Faixa_80_84', 'Faixa_85_89', 'Faixa_90_94', 'Faixa_95_99', 'Faixa_100_plus']
    df_indicadores['Pop_Jovem'] = df_indicadores[jovem_cols].sum(axis=1)
    df_indicadores['Pop_Ativa'] = df_indicadores[ativa_cols].sum(axis=1)
    df_indicadores['Pop_Idosa'] = df_indicadores[idosa_cols].sum(axis=1)
    df_indicadores['Razao_Dependencia'] = (df_indicadores['Pop_Jovem'] + df_indicadores['Pop_Idosa']) / df_indicadores['Pop_Ativa']
    df_indicadores['Indice_Envelhecimento'] = df_indicadores['Pop_Idosa'] / df_indicadores['Pop_Jovem']
    # >> Indicadores de Infraestrutura
    san_precaria_cols = ['Fossa_Rud_Bur', 'Vala', 'Rio_Lag_Cor_Mar', 'Outra', 'Nem_Ban_San']
    df_indicadores['Precariedade_San'] = df_indicadores[san_precaria_cols].sum(axis=1)
    lixo_precario_cols = ['Queimado_Prop', 'Enterrado_Prop', 'Jogado_TerBal_Enc_AP', 'Outro_Destino']
    df_indicadores['Precariedade_Lixo'] = df_indicadores[lixo_precario_cols].sum(axis=1)
    # >> Indicadores de Desigualdade
    df_indicadores['Gap_Alfabetizacao_Racial'] = df_indicadores['Branca_25_34'] - df_indicadores['Preta_25_34']
    df_indicadores['Gap_Alfabetizacao_Geracional'] = df_indicadores['Branca_25_34'] - df_indicadores['Branca_64_plus']
except Exception as e: print(f"Aviso ao calcular indicadores: {e}")
df_indicadores.replace([np.inf, -np.inf], np.nan, inplace=True)
print("-> Criação de indicadores concluída.")

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 3. PRÉ-PROCESSAMENTO
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#%% Preparação e Padronização dos Dados para Clusterização

# >> Seleção dos 18 indicadores-chave criados
features_selecionadas = [
    'Desp_AS_PC', 'Desp_Edu_PC', 'Desp_GA_PC', 'Desp_SS_PC', 'Desp_SP_PC', 'Desp_UH_PC',
    'Prioridade_AS', 'Prioridade_Edu', 'Prioridade_GA', 'Prioridade_SS', 'Prioridade_SP', 'Prioridade_UH',
    'Razao_Dependencia', 'Indice_Envelhecimento', 'Precariedade_San', 'Precariedade_Lixo',
    'Gap_Alfabetizacao_Racial', 'Gap_Alfabetizacao_Geracional'
]
df_cluster = df_indicadores[features_selecionadas].copy()

# >> Tratamento de valores ausentes (NaN) com a média
imputer = SimpleImputer(strategy='mean')
df_cluster_imputed = imputer.fit_transform(df_cluster)

# >> Padronização das variáveis (Z-Score)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster_imputed)

#%% >> Método do Cotovelo para identificação do nº de clusters (K-Means)

wcss = [] # Within-Cluster Sum of Squares
I = range(2, 11) # Testando de 2 a 10 clusters

for i in I:
    kmeans = KMeans(n_clusters=i, init='random', random_state=42, n_init='auto').fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(I, wcss, color='blue', marker='o')
plt.xlabel('Nº de Clusters (k)', fontsize=16)
plt.ylabel('WCSS (Inertia)', fontsize=16)
plt.title('Método do Cotovelo (Elbow Method)', fontsize=16)
plt.grid(True)
plt.show()

#%% >> Método da Silhueta para identificação do nº de clusters (K-Means)

silhueta = []
I = range(2, 11) # Testando de 2 a 10 clusters

for i in I:
    kmeans = KMeans(n_clusters=i, init='random', random_state=42, n_init='auto').fit(df_scaled)
    silhueta.append(silhouette_score(df_scaled, kmeans.labels_))

plt.figure(figsize=(10, 6))
plt.plot(I, silhueta, color='purple', marker='o')
plt.xlabel('Nº de Clusters (k)', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.grid(True)
plt.show()

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 4. MODELAGEM DE CLUSTERS
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#%% >> Método K-Means
# O número de clusters k=4 foi definido com base no Método do Cotovelo e Silhueta
k_ideal = 4
kmeans = KMeans(n_clusters=k_ideal, random_state=42, n_init='auto').fit(df_scaled)

# --> Adicionando a variável de cluster ao banco de dados principal
df_indicadores['cluster_kmeans'] = kmeans.labels_
df_indicadores['cluster_kmeans'] = df_indicadores['cluster_kmeans'].astype('category')

print("-> K-Means executado")

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 5. ANÁLISE E VALIDAÇÃO
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#%% >> Análise de Outliers (DBSCAN)

# O DBSCAN foi utilizado conceitualmente para interpretar o cluster anômalo (outliers)
min_samples = 2 * len(features_selecionadas) # (2 * 18 = 36)
eps = 4.5 # Valor calibrado para este conjunto de dados via análise k-distance

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df_indicadores['cluster_dbscan'] = dbscan.fit_predict(df_scaled)

contagem_kmeans = df_indicadores['cluster_kmeans'].value_counts()
cluster_pequeno_id = contagem_kmeans.idxmin() # Identifica o cluster com menos membros
municipios_outliers_kmeans = df_indicadores[df_indicadores['cluster_kmeans'] == cluster_pequeno_id]
outliers_dbscan_ids = set(df_indicadores[df_indicadores['cluster_dbscan'] == -1].index)
outliers_kmeans_ids = set(municipios_outliers_kmeans.index)
intersecao = len(outliers_kmeans_ids.intersection(outliers_dbscan_ids))

print("\n--- Validndo com o DBSCAN ---")
print(f"O K-Means isolou {len(municipios_outliers_kmeans)} municípios no cluster {cluster_pequeno_id} ('Fora da Curva').")
print(f"O DBSCAN identificou {(df_indicadores['cluster_dbscan'] == -1).sum()} municípios como ruído (-1).")
print(f" -> {intersecao / len(municipios_outliers_kmeans):.1%} dos municípios do cluster 'Fora da Curva' são consistentes com os outliers do DBSCAN.")

#%% >> Análise de Variância (ANOVA)

print("\n--- ANOVA para Validação dos Clusters K-Means ---")
variaveis_anova = ['IFDM_2022', 'Rec_2022', 'Pop_Total', 'Desp_Edu_PC', 'Desp_SS_PC', 'Razao_Dependencia', 'Precariedade_San']

# Tabela para armazenar os resultados da ANOVA
resultados_anova_df = pd.DataFrame()

for var in variaveis_anova:
    anova = pg.anova(dv=var, between='cluster_kmeans', data=df_indicadores, detailed=True)
    # Extraindo a linha relevante (do cluster)
    resultado_var = anova.loc[0, ['Source', 'F', 'p-unc']]
    resultado_var['Variável'] = var
    # Adicionando ao dataframe de resultados
    resultados_anova_df = pd.concat([resultados_anova_df, resultado_var.to_frame().T], ignore_index=True)

# Reorganizando as colunas e exibindo a tabela ANOVA
resultados_anova_df = resultados_anova_df[['Variável', 'F', 'p-unc']]
resultados_anova_df.rename(columns={'F': 'Estatística F', 'p-unc': 'Valor-p'}, inplace=True)
resultados_anova_df['Significativo (p < 0,05)?'] = np.where(resultados_anova_df['Valor-p'] < 0.05, 'Sim', 'Não')
print(resultados_anova_df.to_string(index=False))

#%% >> Identificação das Características dos Clusters (Personas)

caracteristicas_clusters = df_indicadores.groupby('cluster_kmeans')[features_selecionadas + ['IFDM_2022']]
perfil_clusters = caracteristicas_clusters.describe().T
print("\n--- Perfil dos Clusters (Estatísticas Descritivas) ---")
print(perfil_clusters)

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 6. VISUALIZAÇÃO E EXPORTAÇÃO
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#%% >> Visualização dos Clusters com PCA

# Redução de dimensionalidade com PCA para visualização no gráfico
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Adicionando os componentes principais ao dataframe
df_indicadores['PC1'] = df_pca[:, 0]
df_indicadores['PC2'] = df_pca[:, 1]

# Plotando o gráfico
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_indicadores, x='PC1', y='PC2', hue='cluster_kmeans', palette='viridis', s=50, alpha=0.8)

plt.title('Segmentação de Municípios (Baseado em Indicadores Socioeconômicos)', fontsize=16)
plt.xlabel('Primeiro Componente Principal (PC1)', fontsize=12)
plt.ylabel('Segundo Componente Principal (PC2)', fontsize=12)
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

#%% >> Visualização do Boxplot do IFDM

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_indicadores, x='cluster_kmeans', y='IFDM_2022', hue='cluster_kmeans', palette='viridis')
plt.title('Distribuição do IFDM por Cluster', fontsize=16, weight='bold')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('IFDM 2022', fontsize=12)
plt.show()

#%% >> Exportação Final de Relatórios Detalhados em Excel

print("\n--- Gerando Relatórios Completos em Excel ---")

output_filename = 'relatorio_clusters_com_indicadores.xlsx'
print(f"\nExportando relatório principal para '{output_filename}'...")
try:
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df_indicadores.sort_values('cluster_kmeans').to_excel(writer, sheet_name='Dados_Com_Indicadores', index=False)
        
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=features_selecionadas)
        centroids_df['contagem_municipios'] = df_indicadores['cluster_kmeans'].value_counts().sort_index().values
        centroids_df.T.to_excel(writer, sheet_name='Perfil_dos_Clusters_(Centroides)')
            
    print(f"-> Relatório principal '{output_filename}' criado com sucesso.")
except Exception as e:
    print(f"Ocorreu um erro ao salvar o arquivo Excel principal: {e}")

#%% >> Exportação Detalhada por Cluster

print("\n--- Gerando Relatórios por Cluster ---")
output_detalhado = 'municipios_por_cluster.xlsx'
colunas_para_exportar = [
    'Municipio', 'Pop_Total', 'IFDM_2022', 'Rec_2022',
    'Desp_Edu_PC', 'Desp_SS_PC', 'Precariedade_San',
    'Razao_Dependencia', 'Indice_Envelhecimento'
]
colunas_para_exportar = [c for c in colunas_para_exportar if c in df_indicadores.columns]

try:
    with pd.ExcelWriter(output_detalhado, engine='openpyxl') as writer:
        ids_dos_clusters = sorted(df_indicadores['cluster_kmeans'].cat.categories)
        print(f"Gerando abas para os clusters: {ids_dos_clusters}")
        
        for cluster_id in ids_dos_clusters:
            df_filtrado = df_indicadores[df_indicadores['cluster_kmeans'] == cluster_id]
            df_para_exportar = df_filtrado[colunas_para_exportar].sort_values(by='IFDM_2022', ascending=False)
            nome_da_aba = f'Cluster_{cluster_id}'
            df_para_exportar.to_excel(writer, sheet_name=nome_da_aba, index=False)
            print(f"-> Aba '{nome_da_aba}' com {len(df_para_exportar)} municípios criada.")
            
    print(f"-> Relatório '{output_detalhado}' criado com sucesso.")
except Exception as e:
    print(f"Ocorreu um erro ao salvar o arquivo Excel detalhado: {e}")
print("\n--- FIM DO SCRIPT ---")
