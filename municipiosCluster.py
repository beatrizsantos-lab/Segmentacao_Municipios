"""
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Análise de Cluster - SEGMENTAÇÃO SOCIOECONÔMICA DOS MUNICÍPIOS BRASILEIROS
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Objetivo
--------
Este script consolida bases públicas municipais, constrói indicadores 
socioeconômicos e fiscal-orçamentários e aplica o algoritmo K-Means para 
classificar os municípios brasileiros em agrupamentos empíricos.

Autora: Beatriz Santos

Data de criação: 10/06/2025
Última Modificação: 05/06/2026
Versão: 6.0 - Organização do código para melhor replicabilidade e mudança 
              da tabela de Receita Corrente para Receita Orçamentária 

Observações metodológicas
-------------------------
1. As variáveis são imputadas pela média e padronizadas por Z-Score antes
   da clusterização.
2. O K-Means define a classificação final. O DBSCAN é usado apenas como
   validação complementar para observações atípicas.

Como executar
-------------
1. Coloque as planilhas brutas na pasta "Bases de Dados" ou altere a variável
   CAMINHO_BASES no Bloco 01.
2. Execute o script completo ou rode os blocos #%% sequencialmente em uma IDE
   como Spyder, VS Code ou Jupyter.
3. Os arquivos gerados serão salvos na pasta "Resultados".
"""
#%% 00 - IMPORTAÇÃO DE PACOTES

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

import pingouin as pg


#%% 01 - CONFIGURAÇÕES GERAIS DO PROJETO

# Pasta com as bases originais.
# Exemplo esperado: ./Bases de Dados/PopulacaoTotal.xlsx
CAMINHO_BASES = Path("Bases de Dados")

# Pasta de saída dos resultados.
PASTA_RESULTADOS = Path("Resultados")
PASTA_TABELAS = PASTA_RESULTADOS / "tabelas"
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"
PASTA_QGIS = PASTA_RESULTADOS / "qgis"

# Arquivos de entrada utilizados na consolidação.
ARQUIVOS_EXCEL = [
    "Alfabetizacao2022.xlsx",
    "Despesa_AssistSocial_Prev.xlsx",
    "Despesa_Edu_Cult.xlsx",
    "Despesa_GestaoAmbAgric_Org.xlsx",
    "Despesa_Sau_Sanea.xlsx",
    "Despesa_SegPub_Prev.xlsx",
    "Despesa_Urban_Hab.xlsx",
    "Ifdm.xlsx",
    "Lixo2022.xlsx",
    "PopulacaoPorCor2022.xlsx",
    "PopulacaoPorIdade2022.xlsx",
    "PopulacaoTotal.xlsx",
    "ReceitaOrcBruta.xlsx",
    "Saneamento2022.xlsx",
]

# Base-mestra usada no merge. O uso de uma base-mestra reduz o risco de criar
# municípios sem correspondência adequada nas junções.
ARQUIVO_BASE = "PopulacaoTotal.xlsx"
COLUNA_CHAVE = "Cod"
COLUNA_MUNICIPIO = "Municipio"

# Indicadores finais utilizados na análise de agrupamentos.
FEATURES_SELECIONADAS = [
    "Desp_AS_PC",
    "Desp_Edu_PC",
    "Desp_GA_PC",
    "Desp_SS_PC",
    "Desp_SP_PC",
    "Desp_UH_PC",
    "Prioridade_AS",
    "Prioridade_Edu",
    "Prioridade_GA",
    "Prioridade_SS",
    "Prioridade_SP",
    "Prioridade_UH",
    "Razao_Dependencia",
    "Indice_Envelhecimento",
    "Precariedade_San",
    "Precariedade_Lixo",
    "Gap_Alfabetizacao_Racial",
    "Gap_Alfabetizacao_Geracional",
]

# Variáveis usadas para caracterizar e comparar os perfis.
VARIAVEIS_PERFIL = [
    "Pop_Total",
    "IFDM_2022",
    "Rec_2022",
    "Desp_Edu_PC",
    "Desp_SS_PC",
    "Precariedade_San",
    "Razao_Dependencia",
    "Indice_Envelhecimento",
]

# Variáveis utilizadas na ANOVA.
VARIAVEIS_ANOVA = [
    "IFDM_2022",
    "Rec_2022",
    "Pop_Total",
    "Desp_Edu_PC",
    "Desp_SS_PC",
    "Razao_Dependencia",
    "Precariedade_San",
]

# Configurações da modelagem.
K_RANGE = range(2, 11)
K_FINAL = 4
K_COMPARACAO = [4, 5, 6]

# Parâmetros únicos do K-Means para todas as etapas.
# A mesma configuração será usada no cotovelo, na silhueta, nas comparações entre k e no modelo final, 
# evitando divergências entre os resultados exploratórios e a solução final adotada.
KMEANS_PARAMS = {
    "init": "k-means++",
    "random_state": 42,
    "n_init": 50,
}

# Parâmetros do DBSCAN usado como validação complementar.
DBSCAN_EPS = 4.5
DBSCAN_MIN_SAMPLES_MULTIPLIER = 2


#%% 02 - FUNÇÕES AUXILIARES DE ORGANIZAÇÃO E VALIDAÇÃO


def criar_pastas_resultados() -> None:
    """Cria as pastas de saída caso elas ainda não existam."""
    for pasta in [PASTA_RESULTADOS, PASTA_TABELAS, PASTA_GRAFICOS, PASTA_QGIS]:
        pasta.mkdir(parents=True, exist_ok=True)


def padronizar_coluna_codigo(df: pd.DataFrame, coluna_chave: str = COLUNA_CHAVE) -> pd.DataFrame:
    """
    Garante que a coluna do código municipal tenha nome padronizado.

    Algumas bases podem vir com a coluna nomeada como "Código", "Codigo" ou
    variações semelhantes. Esta função renomeia a primeira opção encontrada para
    o nome definido em COLUNA_CHAVE.
    """
    df = df.copy()
    candidatas = [coluna_chave, "Código", "Codigo", "COD", "cod"]

    coluna_encontrada = next((col for col in candidatas if col in df.columns), None)

    if coluna_encontrada is None:
        raise KeyError(
            f"Nenhuma coluna de código municipal foi encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    if coluna_encontrada != coluna_chave:
        df = df.rename(columns={coluna_encontrada: coluna_chave})

    return df


def limpar_codigo_ibge(df: pd.DataFrame, coluna: str = COLUNA_CHAVE) -> pd.DataFrame:
    """Padroniza o código municipal do IBGE como texto de 7 dígitos."""
    df = padronizar_coluna_codigo(df, coluna).copy()

    df[coluna] = (
        df[coluna]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
    )

    # Mantém apenas códigos municipais válidos com 7 dígitos.
    df = df[df[coluna].str.match(r"^\d{7}$", na=False)].copy()
    return df


def validar_colunas(df: pd.DataFrame, colunas: list[str], contexto: str) -> None:
    """Interrompe a execução caso alguma coluna obrigatória esteja ausente."""
    ausentes = [col for col in colunas if col not in df.columns]
    if ausentes:
        raise KeyError(f"Colunas ausentes em {contexto}: {ausentes}")


def ler_planilha(caminho_arquivo: Path, coluna_chave: str = COLUNA_CHAVE) -> pd.DataFrame:
    """Lê uma planilha Excel e padroniza a coluna de código municipal."""
    if not caminho_arquivo.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    df = pd.read_excel(caminho_arquivo, dtype={coluna_chave: str})
    df = limpar_codigo_ibge(df, coluna_chave)
    return df


def salvar_figura(nome_arquivo: str) -> None:
    """Salva a figura atual na pasta de gráficos em PNG."""
    caminho_saida = PASTA_GRAFICOS / nome_arquivo
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")


#%% 03 - FUNÇÕES DE CONSOLIDAÇÃO E ENGENHARIA DE INDICADORES


def consolidar_bases() -> pd.DataFrame:
    """
    Consolida as bases municipais a partir de uma base-mestra.

    Todas as demais bases são agregadas por LEFT JOIN, preservando os municípios
    da base PopulacaoTotal.xlsx e evitando a inclusão de observações sem vínculo
    com a base municipal principal.
    """
    if ARQUIVO_BASE not in ARQUIVOS_EXCEL:
        raise ValueError(f"O arquivo-base '{ARQUIVO_BASE}' precisa estar em ARQUIVOS_EXCEL.")

    print("\n--- Consolidação das bases municipais ---")

    df_consolidado = ler_planilha(CAMINHO_BASES / ARQUIVO_BASE)
    validar_colunas(df_consolidado, [COLUNA_CHAVE, COLUNA_MUNICIPIO], "base-mestra")

    # Remove municípios sem nome e eventuais duplicatas.
    df_consolidado[COLUNA_MUNICIPIO] = df_consolidado[COLUNA_MUNICIPIO].replace(
        r"^\s*$", np.nan, regex=True
    )
    df_consolidado = df_consolidado.dropna(subset=[COLUNA_MUNICIPIO])
    df_consolidado = df_consolidado.drop_duplicates(subset=[COLUNA_CHAVE], keep="first")

    for arquivo in ARQUIVOS_EXCEL:
        if arquivo == ARQUIVO_BASE:
            continue

        df_temp = ler_planilha(CAMINHO_BASES / arquivo)
        df_temp = df_temp.drop_duplicates(subset=[COLUNA_CHAVE], keep="first")

        # Evita colunas duplicadas no merge, mantendo apenas a chave.
        colunas_repetidas = [
            col for col in df_temp.columns if col in df_consolidado.columns and col != COLUNA_CHAVE
        ]
        if colunas_repetidas:
            df_temp = df_temp.drop(columns=colunas_repetidas)

        df_consolidado = pd.merge(
            df_consolidado,
            df_temp,
            on=COLUNA_CHAVE,
            how="left",
        )

    # Converte variáveis numéricas sem alterar código e nome do município.
    for col in df_consolidado.columns:
        if col not in [COLUNA_CHAVE, COLUNA_MUNICIPIO]:
            df_consolidado[col] = pd.to_numeric(df_consolidado[col], errors="coerce")

    print(f"Base consolidada: {df_consolidado.shape[0]} linhas e {df_consolidado.shape[1]} colunas.")
    print(f"Municípios únicos: {df_consolidado[COLUNA_CHAVE].nunique()}")
    return df_consolidado


def criar_indicadores(df_consolidado: pd.DataFrame) -> pd.DataFrame:
    """Cria os indicadores usados na análise de agrupamentos."""
    print("\n--- Engenharia de indicadores ---")
    df = df_consolidado.copy()

    # Indicadores financeiros: despesas per capita e prioridades orçamentárias relativas.
    # As prioridades são calculadas como razão entre despesa empenhada na função e
    # receita orçamentária bruta. Para apresentar em porcentagem, multiplicar por 100.
    despesas_2022 = {
        "AS": "DespAS_2022",
        "Edu": "DespEdu_2022",
        "GA": "DespGA_2022",
        "SS": "DespSS_2022",
        "SP": "DespSP_2022",
        "UH": "DespUH_2022",
    }

    validar_colunas(df, ["Pop_Total", "Rec_2022"], "indicadores financeiros")

    for sigla, coluna_despesa in despesas_2022.items():
        validar_colunas(df, [coluna_despesa], f"despesa {sigla}")
        df[f"Desp_{sigla}_PC"] = df[coluna_despesa] / df["Pop_Total"]
        df[f"Prioridade_{sigla}"] = df[coluna_despesa] / df["Rec_2022"]

    # Indicadores demográficos.
    jovem_cols = ["Faixa_0_4", "Faixa_5_9", "Faixa_10_14"]
    ativa_cols = [
        "Faixa_15_19",
        "Faixa_20_24",
        "Faixa_25_29",
        "Faixa_30_34",
        "Faixa_35_39",
        "Faixa_40_44",
        "Faixa_45_49",
        "Faixa_50_54",
        "Faixa_55_59",
        "Faixa_60_64",
    ]
    idosa_cols = [
        "Faixa_65_69",
        "Faixa_70_74",
        "Faixa_75_79",
        "Faixa_80_84",
        "Faixa_85_89",
        "Faixa_90_94",
        "Faixa_95_99",
        "Faixa_100_plus",
    ]

    validar_colunas(df, jovem_cols + ativa_cols + idosa_cols, "indicadores demográficos")

    df["Pop_Jovem"] = df[jovem_cols].sum(axis=1)
    df["Pop_Ativa"] = df[ativa_cols].sum(axis=1)
    df["Pop_Idosa"] = df[idosa_cols].sum(axis=1)
    df["Razao_Dependencia"] = (df["Pop_Jovem"] + df["Pop_Idosa"]) / df["Pop_Ativa"]
    df["Indice_Envelhecimento"] = df["Pop_Idosa"] / df["Pop_Jovem"]

    # Indicadores de infraestrutura básica.
    san_precaria_cols = ["Fossa_Rud_Bur", "Vala", "Rio_Lag_Cor_Mar", "Outra", "Nem_Ban_San"]
    lixo_precario_cols = ["Queimado_Prop", "Enterrado_Prop", "Jogado_TerBal_Enc_AP", "Outro_Destino"]

    validar_colunas(df, san_precaria_cols, "indicador de precariedade sanitária")
    validar_colunas(df, lixo_precario_cols, "indicador de precariedade do lixo")

    df["Precariedade_San"] = df[san_precaria_cols].sum(axis=1)
    df["Precariedade_Lixo"] = df[lixo_precario_cols].sum(axis=1)

    # Indicadores de desigualdade educacional.
    validar_colunas(
        df,
        ["Branca_25_34", "Preta_25_34", "Branca_64_plus"],
        "indicadores de alfabetização",
    )

    df["Gap_Alfabetizacao_Racial"] = df["Branca_25_34"] - df["Preta_25_34"]
    df["Gap_Alfabetizacao_Geracional"] = df["Branca_25_34"] - df["Branca_64_plus"]

    # Remove valores infinitos gerados por divisões por zero.
    df = df.replace([np.inf, -np.inf], np.nan)

    print("Indicadores criados com sucesso.")
    return df


def limpar_base_modelagem(df_indicadores: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros finais antes da modelagem."""
    print("\n--- Limpeza final da base municipal ---")
    df = limpar_codigo_ibge(df_indicadores, COLUNA_CHAVE)

    df[COLUNA_MUNICIPIO] = df[COLUNA_MUNICIPIO].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=[COLUNA_CHAVE, COLUNA_MUNICIPIO])
    df = df.drop_duplicates(subset=[COLUNA_CHAVE], keep="first")

    print(f"Base válida para modelagem: {df.shape[0]} municípios.")
    print(f"Códigos únicos: {df[COLUNA_CHAVE].nunique()}")
    return df


#%% 04 - FUNÇÕES DE PRÉ-PROCESSAMENTO E ESCOLHA DE K


def preparar_matriz_cluster(df_indicadores: pd.DataFrame):
    """Seleciona, imputa e padroniza as variáveis usadas no K-Means."""
    validar_colunas(df_indicadores, FEATURES_SELECIONADAS, "features de clusterização")

    df_cluster = df_indicadores[FEATURES_SELECIONADAS].copy()

    colunas_vazias = df_cluster.columns[df_cluster.isna().all()].tolist()
    if colunas_vazias:
        raise ValueError(f"As seguintes features estão totalmente vazias: {colunas_vazias}")

    imputer = SimpleImputer(strategy="mean")
    df_cluster_imputed = imputer.fit_transform(df_cluster)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cluster_imputed)

    return df_cluster, df_cluster_imputed, df_scaled, imputer, scaler


def calcular_metricas_k(df_scaled: np.ndarray) -> pd.DataFrame:
    """Calcula métricas de validação interna para diferentes valores de k."""
    metricas = []

    for k in K_RANGE:
        modelo = KMeans(n_clusters=k, **KMEANS_PARAMS)
        labels = modelo.fit_predict(df_scaled)

        metricas.append(
            {
                "k": k,
                "inercia": modelo.inertia_,
                "silhueta": silhouette_score(df_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(df_scaled, labels),
                "davies_bouldin": davies_bouldin_score(df_scaled, labels),
                "menor_cluster": pd.Series(labels).value_counts().min(),
                "maior_cluster": pd.Series(labels).value_counts().max(),
            }
        )

    metricas_df = pd.DataFrame(metricas)
    return metricas_df


def plotar_elbow(metricas_df: pd.DataFrame) -> None:
    """Gera e salva o gráfico do método do cotovelo."""
    plt.figure(figsize=(10, 6))
    plt.plot(metricas_df["k"], metricas_df["inercia"], marker="o")
    plt.xlabel("Número de clusters (k)", fontsize=12)
    plt.ylabel("Inércia", fontsize=12)
    plt.grid(True)
    salvar_figura("grafico_elbow_kmeans.png")
    plt.show()


def plotar_silhueta(metricas_df: pd.DataFrame) -> None:
    """
    Gera e salva o gráfico de silhueta a partir de k=3.

    O k=2 é calculado e mantido na tabela, mas não é plotado porque tende a
    produzir uma separação excessivamente agregada e pouco informativa para a
    construção da tipologia municipal.
    """
    metricas_plot = metricas_df[metricas_df["k"] >= 3].copy()

    plt.figure(figsize=(10, 6))
    plt.plot(metricas_plot["k"], metricas_plot["silhueta"], marker="o")
    plt.xlabel("Número de clusters (k)", fontsize=12)
    plt.ylabel("Coeficiente médio de silhueta", fontsize=12)
    plt.grid(True)
    salvar_figura("grafico_silhueta_kmeans_k3_k10.png")
    plt.show()


def comparar_solucoes_k(df_indicadores: pd.DataFrame, df_scaled: np.ndarray):
    """Compara soluções candidatas de k e resume seus perfis médios."""
    comparacao = []
    perfis = []

    variaveis_disponiveis = [v for v in VARIAVEIS_PERFIL if v in df_indicadores.columns]

    for k in K_COMPARACAO:
        modelo = KMeans(n_clusters=k, **KMEANS_PARAMS)
        labels = modelo.fit_predict(df_scaled)

        comparacao.append(
            {
                "k": k,
                "silhueta": silhouette_score(df_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(df_scaled, labels),
                "davies_bouldin": davies_bouldin_score(df_scaled, labels),
                "inercia": modelo.inertia_,
                "menor_cluster": pd.Series(labels).value_counts().min(),
                "maior_cluster": pd.Series(labels).value_counts().max(),
            }
        )

        df_temp = df_indicadores.copy()
        df_temp[f"cluster_k{k}"] = labels

        resumo = (
            df_temp.groupby(f"cluster_k{k}")[variaveis_disponiveis]
            .mean()
            .reset_index()
        )
        resumo.insert(0, "k", k)
        resumo = resumo.rename(columns={f"cluster_k{k}": "cluster"})
        perfis.append(resumo)

        print(f"\n--- Perfil médio dos clusters para k={k} ---")
        print(resumo)
        print("\nContagem por cluster:")
        print(df_temp[f"cluster_k{k}"].value_counts().sort_index())

    comparacao_df = pd.DataFrame(comparacao)
    perfis_df = pd.concat(perfis, ignore_index=True)
    return comparacao_df, perfis_df


def comparar_k4_k5(df_indicadores: pd.DataFrame, df_scaled: np.ndarray):
    """Gera tabelas auxiliares de comparação entre as soluções k=4 e k=5."""
    modelo4 = KMeans(n_clusters=4, **KMEANS_PARAMS)
    modelo5 = KMeans(n_clusters=5, **KMEANS_PARAMS)

    labels4 = modelo4.fit_predict(df_scaled)
    labels5 = modelo5.fit_predict(df_scaled)

    df_compara = df_indicadores[[COLUNA_CHAVE, COLUNA_MUNICIPIO]].copy()
    df_compara["cluster_k4"] = labels4
    df_compara["cluster_k5"] = labels5

    tabela_cruzada = pd.crosstab(df_compara["cluster_k4"], df_compara["cluster_k5"])

    # Extração de UF a partir do padrão "Município (UF)", quando disponível.
    df_compara["UF"] = df_compara[COLUNA_MUNICIPIO].astype(str).str.extract(r"\(([A-Z]{2})\)$")

    return df_compara, tabela_cruzada


#%% 05 - FUNÇÕES DE MODELAGEM, VALIDAÇÃO E CARACTERIZAÇÃO


def executar_kmeans_final(df_indicadores: pd.DataFrame, df_scaled: np.ndarray):
    """Executa o K-Means final e adiciona o cluster ao dataframe."""
    modelo = KMeans(n_clusters=K_FINAL, **KMEANS_PARAMS).fit(df_scaled)

    df_resultado = df_indicadores.copy()
    df_resultado["cluster_kmeans"] = pd.Categorical(modelo.labels_)

    print("\n--- K-Means final executado ---")
    print(df_resultado["cluster_kmeans"].value_counts().sort_index())
    print(f"Total de municípios classificados: {df_resultado.shape[0]}")

    return df_resultado, modelo


def validar_com_dbscan(df_indicadores: pd.DataFrame, df_scaled: np.ndarray) -> pd.DataFrame:
    """
    Aplica DBSCAN como validação complementar do menor cluster do K-Means.

    O DBSCAN não substitui a tipologia final. Ele apenas verifica se o menor
    agrupamento do K-Means também aparece como conjunto de observações atípicas
    em uma abordagem baseada em densidade.
    """
    min_samples = DBSCAN_MIN_SAMPLES_MULTIPLIER * len(FEATURES_SELECIONADAS)
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=min_samples)

    df_indicadores["validacao_dbscan"] = dbscan.fit_predict(df_scaled)

    contagem_kmeans = df_indicadores["cluster_kmeans"].value_counts().sort_index()
    cluster_pequeno_id = contagem_kmeans.idxmin()
    n_cluster_pequeno = contagem_kmeans.loc[cluster_pequeno_id]

    municipios_cluster_pequeno = df_indicadores[
        df_indicadores["cluster_kmeans"] == cluster_pequeno_id
    ]

    outliers_dbscan_ids = set(df_indicadores[df_indicadores["validacao_dbscan"] == -1].index)
    outliers_kmeans_ids = set(municipios_cluster_pequeno.index)
    intersecao = len(outliers_kmeans_ids.intersection(outliers_dbscan_ids))

    percentual_validado = intersecao / n_cluster_pequeno if n_cluster_pequeno > 0 else np.nan

    resumo = pd.DataFrame(
        {
            "Indicador": [
                "Menor cluster do K-Means",
                "Municípios classificados como ruído no DBSCAN",
                "Interseção entre menor cluster e ruído DBSCAN",
                "Percentual do menor cluster validado pelo DBSCAN",
            ],
            "Resultado": [
                n_cluster_pequeno,
                int((df_indicadores["validacao_dbscan"] == -1).sum()),
                intersecao,
                percentual_validado,
            ],
        }
    )

    print("\n--- Validação complementar com DBSCAN ---")
    print(
        f"O K-Means isolou {n_cluster_pequeno} municípios no cluster {cluster_pequeno_id}, "
        "o menor agrupamento da solução."
    )
    print(
        f"O DBSCAN identificou {(df_indicadores['validacao_dbscan'] == -1).sum()} "
        "municípios como ruído (-1)."
    )
    print(
        f"{percentual_validado:.1%} dos municípios do menor cluster do K-Means "
        "também foram identificados como ruído pelo DBSCAN."
    )

    return resumo


def executar_anova(df_indicadores: pd.DataFrame) -> pd.DataFrame:
    """Executa ANOVA para verificar diferenças médias entre os clusters."""
    print("\n--- ANOVA para validação dos clusters K-Means ---")

    variaveis_validas = [
        v for v in VARIAVEIS_ANOVA if v in df_indicadores.columns and df_indicadores[v].notna().sum() > 0
    ]

    resultados = []
    for variavel in variaveis_validas:
        try:
            anova = pg.anova(dv=variavel, between="cluster_kmeans", data=df_indicadores, detailed=True)
            linha = anova.loc[0, ["Source", "F", "p-unc"]].to_dict()
            linha["Variável"] = variavel
            resultados.append(linha)
        except Exception as erro:
            print(f"Aviso: não foi possível executar ANOVA para {variavel}: {erro}")

    resultados_df = pd.DataFrame(resultados)

    if resultados_df.empty:
        print("Nenhum resultado de ANOVA foi gerado.")
        return resultados_df

    resultados_df = resultados_df[["Variável", "F", "p-unc"]].rename(
        columns={"F": "Estatística F", "p-unc": "Valor-p"}
    )
    resultados_df["Significativo (p < 0,05)?"] = np.where(
        resultados_df["Valor-p"] < 0.05, "Sim", "Não"
    )

    print(resultados_df.to_string(index=False))
    return resultados_df


def caracterizar_clusters(df_indicadores: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas descritivas dos indicadores por cluster."""
    colunas_descritivas = FEATURES_SELECIONADAS.copy()
    if "IFDM_2022" in df_indicadores.columns:
        colunas_descritivas.append("IFDM_2022")

    perfil = df_indicadores.groupby("cluster_kmeans", observed=False)[colunas_descritivas].describe().T

    print("\n--- Perfil dos clusters: estatísticas descritivas ---")
    print(perfil)
    return perfil


def gerar_tabela_medias(df_indicadores: pd.DataFrame) -> pd.DataFrame:
    """Gera tabela de médias dos 18 indicadores por cluster."""
    tabela = df_indicadores.groupby("cluster_kmeans", observed=False)[FEATURES_SELECIONADAS].mean().T
    tabela.columns = [f"Cluster {col}" for col in tabela.columns]

    contagem_clusters = df_indicadores["cluster_kmeans"].value_counts().sort_index()
    linha_contagem = pd.DataFrame(
        [contagem_clusters.values],
        columns=tabela.columns,
        index=["N_municipios"],
    )

    tabela = pd.concat([linha_contagem, tabela]).round(3)

    print("\n--- Médias dos indicadores por cluster ---")
    print(tabela)
    return tabela


#%% 06 - FUNÇÕES DE VISUALIZAÇÃO E EXPORTAÇÃO


def gerar_pca(df_indicadores: pd.DataFrame, df_scaled: np.ndarray):
    """Aplica PCA com dois componentes apenas para visualização dos clusters."""
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(df_scaled)

    df_resultado = df_indicadores.copy()
    df_resultado["PC1"] = componentes[:, 0]
    df_resultado["PC2"] = componentes[:, 1]

    variancia = pca.explained_variance_ratio_
    print("\n--- PCA para visualização ---")
    print(f"Variância explicada por PC1: {variancia[0]:.2%}")
    print(f"Variância explicada por PC2: {variancia[1]:.2%}")

    return df_resultado, pca


def plotar_pca(df_indicadores: pd.DataFrame) -> None:
    """
    Gera gráfico de dispersão dos clusters nos dois componentes principais. 
    Foi utilizado na análise com finalidade, apenas, de exploração. 
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_indicadores,
        x="PC1",
        y="PC2",
        hue="cluster_kmeans",
        palette="viridis",
        s=50,
        alpha=0.8,
    )
    plt.xlabel("Primeiro Componente Principal (PC1)", fontsize=12)
    plt.ylabel("Segundo Componente Principal (PC2)", fontsize=12)
    plt.legend(title="Cluster")
    plt.grid(True)
    salvar_figura("grafico_pca_clusters.png")
    plt.show()


def plotar_boxplot_ifdm(df_indicadores: pd.DataFrame) -> None:
    """Gera boxplot do IFDM segundo os clusters."""
    if "IFDM_2022" not in df_indicadores.columns:
        print("Coluna IFDM_2022 ausente. Boxplot não foi gerado.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_indicadores,
        x="cluster_kmeans",
        y="IFDM_2022",
        hue="cluster_kmeans",
        palette="viridis",
    )
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("IFDM 2022", fontsize=12)
    plt.legend(title="Cluster", loc="lower right")
    plt.grid(False)
    salvar_figura("boxplot_ifdm_por_cluster.png")
    plt.show()


def exportar_relatorio_principal(
    df_indicadores: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    resultados_anova_df: pd.DataFrame,
    metricas_escolha_k_df: pd.DataFrame,
    comparacao_k_df: pd.DataFrame,
    perfis_comparacao_df: pd.DataFrame,
    resumo_validacao_dbscan: pd.DataFrame,
) -> None:
    """Exporta relatório principal com bases, centróides, métricas e validações."""
    caminho_saida = PASTA_TABELAS / "relatorio_clusters_com_indicadores.xlsx"

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_df = pd.DataFrame(centroids, columns=FEATURES_SELECIONADAS)
    centroids_df.insert(0, "cluster_kmeans", range(K_FINAL))
    centroids_df["contagem_municipios"] = (
        df_indicadores["cluster_kmeans"].value_counts().sort_index().values
    )

    with pd.ExcelWriter(caminho_saida, engine="openpyxl") as writer:
        df_indicadores.sort_values("cluster_kmeans").to_excel(
            writer, sheet_name="Dados_Com_Indicadores", index=False
        )
        centroids_df.to_excel(writer, sheet_name="Centroides", index=False)

        if not resultados_anova_df.empty:
            resultados_anova_df.to_excel(writer, sheet_name="ANOVA", index=False)

        metricas_escolha_k_df.to_excel(writer, sheet_name="Metricas_k2_k10", index=False)
        comparacao_k_df.to_excel(writer, sheet_name="Comparacao_k", index=False)
        perfis_comparacao_df.to_excel(writer, sheet_name="Perfis_k_comparacao", index=False)
        resumo_validacao_dbscan.to_excel(writer, sheet_name="Validacao_DBSCAN", index=False)

    print(f"Relatório principal salvo em: {caminho_saida}")


def exportar_municipios_por_cluster(df_indicadores: pd.DataFrame) -> None:
    """Exporta um arquivo Excel com uma aba para cada cluster final."""
    caminho_saida = PASTA_TABELAS / "municipios_por_cluster.xlsx"

    colunas_exportar = [
        COLUNA_CHAVE,
        COLUNA_MUNICIPIO,
        "cluster_kmeans",
        "Pop_Total",
        "IFDM_2022",
        "Rec_2022",
        "Desp_Edu_PC",
        "Desp_SS_PC",
        "Precariedade_San",
        "Razao_Dependencia",
        "Indice_Envelhecimento",
    ]
    colunas_exportar = [c for c in colunas_exportar if c in df_indicadores.columns]

    with pd.ExcelWriter(caminho_saida, engine="openpyxl") as writer:
        ids_clusters = sorted(df_indicadores["cluster_kmeans"].cat.categories)

        for cluster_id in ids_clusters:
            df_cluster = df_indicadores[df_indicadores["cluster_kmeans"] == cluster_id]
            df_cluster = df_cluster[colunas_exportar].sort_values(
                by="IFDM_2022", ascending=False
            )
            df_cluster.to_excel(writer, sheet_name=f"Cluster_{cluster_id}", index=False)

    print(f"Relatório por cluster salvo em: {caminho_saida}")


def exportar_qgis(df_indicadores: pd.DataFrame) -> None:
    """Exporta arquivo CSV simplificado para junção no QGIS."""
    colunas_qgis = [
        COLUNA_CHAVE,
        COLUNA_MUNICIPIO,
        "cluster_kmeans",
        "Pop_Total",
        "IFDM_2022",
        "Rec_2022",
        "Desp_Edu_PC",
        "Desp_SS_PC",
        "Precariedade_San",
        "Razao_Dependencia",
        "Indice_Envelhecimento",
    ]
    colunas_qgis = [c for c in colunas_qgis if c in df_indicadores.columns]

    df_qgis = df_indicadores[colunas_qgis].copy()
    df_qgis = limpar_codigo_ibge(df_qgis, COLUNA_CHAVE)

    caminho_saida = PASTA_QGIS / "classificacao_municipios_qgis.csv"
    df_qgis.to_csv(caminho_saida, sep=";", index=False, encoding="utf-8-sig")

    print(f"Arquivo para QGIS salvo em: {caminho_saida}")


def exportar_tabelas_auxiliares(
    tabela_medias_indicadores: pd.DataFrame,
    perfil_clusters: pd.DataFrame,
    metricas_escolha_k_df: pd.DataFrame,
    df_compara_k4_k5: pd.DataFrame,
    tabela_cruzada_k4_k5: pd.DataFrame,
) -> None:
    """Exporta tabelas auxiliares usadas na análise."""
    tabela_medias_indicadores.to_excel(PASTA_TABELAS / "tabela_medias_indicadores_por_cluster.xlsx")
    perfil_clusters.to_excel(PASTA_TABELAS / "estatisticas_descritivas_por_cluster.xlsx")
    metricas_escolha_k_df.to_excel(PASTA_TABELAS / "metricas_escolha_k.xlsx", index=False)
    df_compara_k4_k5.to_excel(PASTA_TABELAS / "comparacao_clusters_k4_k5.xlsx", index=False)
    tabela_cruzada_k4_k5.to_excel(PASTA_TABELAS / "tabela_cruzada_k4_k5.xlsx")

    print("Tabelas auxiliares salvas na pasta de resultados.")


# 07 - EXECUÇÃO
#%% 7.1 - CONSOLIDAÇÃO E INDICADORES

criar_pastas_resultados()

df_consolidado = consolidar_bases()
df_indicadores = criar_indicadores(df_consolidado)
df_indicadores = limpar_base_modelagem(df_indicadores)


#%% 7.2 - PRÉ-PROCESSAMENTO

_, df_cluster_imputed, df_scaled, imputer, scaler = preparar_matriz_cluster(df_indicadores)


#%% 7.3 - ESCOLHA DO NÚMERO DE CLUSTERS

metricas_escolha_k_df = calcular_metricas_k(df_scaled)

print("\n--- Métricas para escolha do número de clusters ---")
print(metricas_escolha_k_df.to_string(index=False))

plotar_elbow(metricas_escolha_k_df)
plotar_silhueta(metricas_escolha_k_df)

comparacao_k_df, perfis_comparacao_df = comparar_solucoes_k(df_indicadores, df_scaled)
df_compara_k4_k5, tabela_cruzada_k4_k5 = comparar_k4_k5(df_indicadores, df_scaled)

print("\n--- Tabela cruzada: k=4 versus k=5 ---")
print(tabela_cruzada_k4_k5)


#%% 7.4 - MODELO FINAL K-MEANS

df_indicadores, kmeans = executar_kmeans_final(df_indicadores, df_scaled)


#%% 7.5 - VALIDAÇÕES E CARACTERIZAÇÃO DOS CLUSTERS

resumo_validacao_dbscan = validar_com_dbscan(df_indicadores, df_scaled)
resultados_anova_df = executar_anova(df_indicadores)
perfil_clusters = caracterizar_clusters(df_indicadores)
tabela_medias_indicadores = gerar_tabela_medias(df_indicadores)


#%% 7.6 - VISUALIZAÇÕES

df_indicadores, pca = gerar_pca(df_indicadores, df_scaled)
plotar_pca(df_indicadores)
plotar_boxplot_ifdm(df_indicadores)


#%% 7.7 - EXPORTAÇÕES FINAIS

exportar_tabelas_auxiliares(
    tabela_medias_indicadores=tabela_medias_indicadores,
    perfil_clusters=perfil_clusters,
    metricas_escolha_k_df=metricas_escolha_k_df,
    df_compara_k4_k5=df_compara_k4_k5,
    tabela_cruzada_k4_k5=tabela_cruzada_k4_k5,
)

exportar_relatorio_principal(
    df_indicadores=df_indicadores,
    scaler=scaler,
    kmeans=kmeans,
    resultados_anova_df=resultados_anova_df,
    metricas_escolha_k_df=metricas_escolha_k_df,
    comparacao_k_df=comparacao_k_df,
    perfis_comparacao_df=perfis_comparacao_df,
    resumo_validacao_dbscan=resumo_validacao_dbscan,
)

exportar_municipios_por_cluster(df_indicadores)
exportar_qgis(df_indicadores)

print("\n--- THAT'S ALL, FOLKS! ---")
