# Segmentação Socioeconômica de Municípios Brasileiros por Análise de Agrupamentos

Este repositório contém o código utilizado para realizar uma análise de agrupamentos dos municípios brasileiros, com base em indicadores fiscais, demográficos, infraestruturais e educacionais. O objetivo é construir uma tipologia empírica dos municípios, permitindo sintetizar a heterogeneidade territorial brasileira e subsidiar diagnósticos públicos mais sensíveis às diferenças locais.

A análise foi desenvolvida originalmente como parte de um trabalho acadêmico no MBA em Data Science & Analytics da USP/ESALQ.

## Objetivo

Aplicar técnicas de análise de agrupamentos para segmentar os municípios brasileiros em perfis socioeconômicos distintos, utilizando dados públicos referentes ao ano de 2022.

A proposta não é definir uma alocação ótima de recursos públicos, mas fornecer subsídios para diagnósticos territoriais e para reflexões sobre políticas públicas baseadas em evidências.

## Metodologia

A metodologia utiliza o algoritmo **K-Means** como técnica principal de agrupamento. O número de clusters foi definido com base na análise conjunta de:

- método do cotovelo;
- coeficiente médio de silhueta;
- métricas auxiliares de validação interna;
- tamanho dos agrupamentos;
- interpretabilidade substantiva dos perfis formados.

Além do K-Means, foram utilizadas análises complementares:

- **DBSCAN**, para verificar a presença de observações atípicas;
- **ANOVA**, para avaliar se os agrupamentos apresentam diferenças estatisticamente significativas em relação aos principais indicadores;
- **PCA**, para visualização bidimensional dos agrupamentos.

## Indicadores utilizados

A análise considera 18 indicadores construídos a partir das bases municipais.

### Indicadores de despesa pública per capita

- Despesa em Assistência Social e Previdência per capita;
- Despesa em Educação e Cultura per capita;
- Despesa em Gestão Ambiental, Agricultura e Organização Agrária per capita;
- Despesa em Saúde e Saneamento per capita;
- Despesa em Segurança Pública e Previdência per capita;
- Despesa em Urbanismo e Habitação per capita.

### Indicadores de prioridade orçamentária relativa

- Prioridade relativa em Assistência Social e Previdência;
- Prioridade relativa em Educação e Cultura;
- Prioridade relativa em Gestão Ambiental, Agricultura e Organização Agrária;
- Prioridade relativa em Saúde e Saneamento;
- Prioridade relativa em Segurança Pública e Previdência;
- Prioridade relativa em Urbanismo e Habitação.

Esses indicadores são calculados pela razão entre a despesa empenhada na função e a receita orçamentária bruta municipal.

### Indicadores demográficos

- Razão de dependência;
- Índice de envelhecimento.

### Indicadores de infraestrutura

- Índice de precariedade sanitária;
- Índice de precariedade na destinação do lixo.

### Indicadores de desigualdade educacional

- Gap de alfabetização racial;
- Gap de alfabetização geracional.

## Bases de dados

O script utiliza arquivos em formato `.xlsx`, armazenados localmente na pasta `Bases de Dados`.

Arquivos esperados:

```text
Bases de Dados/
├── Alfabetizacao2022.xlsx
├── Despesa_AssistSocial_Prev.xlsx
├── Despesa_Edu_Cult.xlsx
├── Despesa_GestaoAmbAgric_Org.xlsx
├── Despesa_Sau_Sanea.xlsx
├── Despesa_SegPub_Prev.xlsx
├── Despesa_Urban_Hab.xlsx
├── Ifdm.xlsx
├── Lixo2022.xlsx
├── PopulacaoPorCor2022.xlsx
├── PopulacaoPorIdade2022.xlsx
├── PopulacaoTotal.xlsx
├── ReceitaOrcBruta.xlsx
└── Saneamento2022.xlsx
```

A base `PopulacaoTotal.xlsx` é utilizada como base-mestra para consolidação dos municípios.

## Estrutura sugerida do projeto

```text
Cluster_Municipios/
├── Bases de Dados/
│   └── arquivos .xlsx utilizados na análise
├── Resultados/
│   └── arquivos gerados pelo script
├── municipiosClusterV6_1_github.py
├── requirements_cluster_municipios.txt
└── README.md
```

## Como executar

1. Clone o repositório:

```bash
git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
```

2. Acesse a pasta do projeto:

```bash
cd NOME-DO-REPOSITORIO
```

3. Crie um ambiente virtual, se desejar:

```bash
python -m venv .venv
```

4. Ative o ambiente virtual.

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/Mac:

```bash
source .venv/bin/activate
```

5. Instale as dependências:

```bash
pip install -r requirements_cluster_municipios.txt
```

6. Certifique-se de que os arquivos `.xlsx` estão dentro da pasta `Bases de Dados`.

7. Execute o script:

```bash
python municipiosCluster.py
```

## Principais saídas geradas

O script cria uma pasta `Resultados`, onde são salvos os arquivos gerados pela análise.

Entre as principais saídas, estão:

```text
Resultados/
├── grafico_elbow_kmeans.png
├── grafico_silhueta_kmeans_k3_k10.png
├── grafico_pca_clusters.png
├── grafico_boxplot_ifdm.png
├── tabela_medias_indicadores_por_cluster.xlsx
├── relatorio_clusters_com_indicadores.xlsx
├── municipios_por_cluster.xlsx
├── classificacao_municipios_qgis.csv
├── comparacao_clusters_k4_k5.xlsx
├── municipios_cluster_4_k5.xlsx
├── municipios_cluster_4_k5_detalhado.xlsx
└── municipios_migraram_brasil_profundo_para_alto_desempenho.xlsx
```

O arquivo `classificacao_municipios_qgis.csv` pode ser utilizado para integração com malhas municipais no QGIS, por meio do código municipal do IBGE.

## Observação sobre os nomes dos clusters

O código mantém os agrupamentos identificados pelo K-Means como rótulos numéricos, por exemplo:

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
```

A nomeação substantiva dos perfis deve ser feita apenas na etapa de interpretação dos resultados, pois pode variar conforme a base de dados, os indicadores utilizados e a solução de agrupamento escolhida.

No estudo original, os agrupamentos foram interpretados como perfis municipais distintos, mas esses nomes não são fixados no código para preservar a reprodutibilidade e a flexibilidade metodológica.

## Validação complementar

### DBSCAN

O DBSCAN é utilizado apenas como validação complementar para identificar observações atípicas. Ele não define a tipologia final dos municípios.

No script, o parâmetro `eps` define o raio de vizinhança no espaço das variáveis padronizadas, enquanto `min_samples` é calculado em função do número de indicadores utilizados.

### ANOVA

A ANOVA é utilizada para verificar se os clusters apresentam diferenças estatisticamente significativas em relação a indicadores selecionados, como IFDM, receita orçamentária, população, despesas per capita e precariedade sanitária.

## Reprodutibilidade

Para garantir maior consistência dos resultados, o algoritmo K-Means utiliza `random_state = 42` e `n_init = 50`.

As variáveis são previamente imputadas pela média e padronizadas por meio do `StandardScaler`, de modo que todas as etapas do K-Means utilizem a mesma base transformada.

## Limitações

A análise utiliza dados referentes ao ano de 2022, portanto representa um retrato específico do período analisado. O método tem caráter descritivo e exploratório, não permitindo estabelecer relações causais entre gasto público e desempenho socioeconômico.

Além disso, os resultados dependem da qualidade, disponibilidade e compatibilidade das bases públicas utilizadas.

## Possíveis extensões

Trabalhos futuros podem ampliar a análise por meio de:

- dados em painel;
- comparação da evolução dos municípios ao longo do tempo;
- incorporação de indicadores de saúde, educação, violência, conectividade digital e capacidade administrativa;
- aplicação de métodos de eficiência;
- uso de econometria espacial;
- modelos preditivos para investigar fatores associados à mudança de perfil municipal.

## Autoria

Beatriz Santos  
Mestranda em Economia Aplicada (PPGE/UFAL)  
MBA em Data Science & Analytics (USP/ESALQ)

Contato: beatrizfsantos@usp.br / beatriz.santos@feac.ufal.br

## Citação

Sugestão (sujeita a alterações):

SANTOS, B.F. S.; THEODORO, R. Segmentação socioeconômica de municípios brasileiros por análise de agrupamentos. Repositório de código e resultados complementares, 2026. Disponível em: <link do repositório>. Acesso em: dia mês ano.

## Licença

Este projeto está licenciado sob a licença MIT - consulte o arquivo [LICENSE](LICENSE)
