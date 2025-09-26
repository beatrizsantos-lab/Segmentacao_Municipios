# Segmentação Socioeconômica de Municípios Brasileiros

Análise de cluster para segmentar os municípios brasileiros com base em características socioeconômicas, visando orientar uma alocação de recursos mais assertiva e equitativa. Este repositório contém o código e a metodologia do Trabalho de Conclusão de Curso do MBA em Data Science & Analytics USP/ESALQ.

**Autora:** Beatriz Santos ([beatrizfsantos@usp.br](mailto:beatrizfsantos@usp.br))

---
### Tabela de Conteúdos
1. [Sobre o Projeto](#-sobre-o-projeto)
2. [Resultados: As 4 Personas dos Municípios](#-resultados-as-4-personas-dos-municípios-brasileiros)
3. [Estrutura do Repositório](#-estrutura-do-repositório)
4. [Metodologia](#-metodologia)
5. [Stack de Tecnologias](#-stack-de-tecnologias)
6. [Como Executar o Projeto](#-como-executar-o-projeto)
7. [Autora e Citação](#-autora-e-citação)
8. [Licença](#-licença)

---

## Sobre o Projeto

A alocação eficiente de recursos públicos é um desafio central para a gestão governamental no Brasil, um país marcado por profundas disparidades regionais. A tomada de decisão baseada em evidências é fundamental para mitigar desigualdades e promover um desenvolvimento socioeconômico equilibrado. Este estudo visa responder a seguinte questão de pesquisa: **como técnicas de “clustering” podem ser aplicadas para otimizar a alocação de recursos públicos, considerando indicadores socioeconômicos extraídos de bases de dados abertas?**

Utilizando técnicas de aprendizado de máquina não supervisionado, o trabalho segmenta 5.596 municípios brasileiros (Censo de 2022) em grupos com perfis socioeconômicos distintos, fornecendo uma ferramenta analítica para a formulação de políticas públicas direcionadas e mais eficazes.

---

## Resultados: As 4 Personas dos Municípios Brasileiros

A análise resultou na identificação de quatro perfis (personas) de municípios, estatisticamente distintos:

- **Cluster 0 — Brasil Profundo (2.212 municípios):** Caracterizado por carências estruturais críticas, principalmente em saneamento básico, e com os menores níveis de investimento per capita em áreas como saúde e educação.
- **Cluster 1 — Fora da Curva (16 municípios):**  Um grupo anômalo com características financeiras atípicas, possivelmente devido a inconsistências nos dados declarados, que requerem análise individualizada.
- **Cluster 2 — Estruturados (2.546 municípios):** Representam o perfil mais equilibrado e com os indicadores de infraestrutura e desenvolvimento mais positivos, como os menores índices de precariedade sanitária e a mediana de IFDM mais elevada.
- **Cluster 3 — Investidores (822 municípios):**  Municípios com alta capacidade de investimento e os maiores gastos per capita em saúde e educação. No entanto, o alto investimento nem sempre se traduz em infraestrutura de qualidade, sugerindo possíveis ineficiências de gestão.

---

## Estrutura do Repositório

```
.
├── Bases de Dados/           # 14 arquivos com dados do IBGE (dados de entrada) .xlsx
├── Dados Gerados/            # Relatórios gerados
├── .gitignore                # Arquivos/dirs a ignorar
├── README.md                 # Este arquivo
└── municipiosCluster.py      # Script
```

---

## Metodologia

A análise foi conduzida como uma pesquisa documental e descritiva com abordagem quantitativa. O fluxo de trabalho implementado no script Python (`municipiosCluster.py`) segue as seguintes etapas:

1. **Coleta e Consolidação de Dados:** Utilização de dados secundários de fontes públicas como IBGE, Ipeadata e Portal da Transparência, com foco no ano de **2022**. Os dados de múltiplos arquivos foram consolidados usando o código do município do IBGE como chave primária.
- 14 planilhas **.xlsx**;  
- Chave primária: **código do município (Cod)**.

2. **Engenharia de Atributos:** Criação de 18 indicadores-chave a partir dos dados brutos (ver a pasta Bases de Dados), divididos em quatro categorias: **Finanças, Demografia, Infraestrutura e Desigualdade**.

3. **Pré-processamento:** Padronização dos 18 indicadores utilizando a técnica Z-score (`StandardScaler`) para garantir que todas as variáveis tivessem a mesma contribuição na modelagem. 
- Seleção de **k** via **Método do Cotovelo** e **Silhueta**.

4. **Modelagem e Validação:**
    - Aplicação do algoritmo **K-Means** para agrupar os municípios. O número de clusters foi definido como K=4 com base no Método do Cotovelo, Análise de Silhueta e, de forma decisiva, na interpretabilidade dos grupos gerados.
    - A lógica do algoritmo **DBSCAN** foi utilizada para interpretar o cluster de outliers identificado pelo K-Means (“Fora da Curva”).
    - A robustez da segmentação foi validada pela **Análise de Variância (ANOVA)**, que confirmou que os clusters eram estatisticamente distintos entre si (p < 0,05).
    - A **Análise de Componentes Principais (PCA)** foi empregada para visualizar os clusters em um gráfico 2D.

---

## Stack de Tecnologias

- **Análise de Dados:** `pandas`, `numpy`  
- **Machine Learning:** `scikit-learn`, `scipy`  
- **Análise Estatística:** `pingouin`  
- **Visualização:** `matplotlib`, `seaborn`, `plotly`  
- **Arquivos:** `os`, `openpyxl`

---

## Como Executar o Projeto

### 1) Pré-requisitos
- **Python 3.8+**  
- **Git**

### 2) Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

# Crie e ative um ambiente virtual
python -m venv venv
# Windows:
# venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 3) Configuração dos Dados

1. Crie a pasta **Clustering** na raiz.  
2. Coloque dentro dela os **14 arquivos .xlsx** de dados brutos.  
3. Ajuste o caminho de dados no início de `municipiosCluster.py` para um **caminho relativo**:

```python
# Altere de:
# caminho_pasta = r'C:\Users\beatr\OneDrive\...'
# Para:
caminho_pasta = 'Clustering'
```

### 4) Execução

- O script é sequencial e organizado em **células (#%%)**.  
- Utilizou-se o **VS Code (modo interativo)**.  
- Rode as células em ordem; os gráficos aparecerão na área de plotagem e os relatórios serão salvos na raiz.

---

## Saídas (Outputs)

Serão gerados:

- **`relatorio_clusters_com_indicadores.xlsx`**  
  - Todos os municípios, indicadores e cluster atribuído.  
  - Aba extra com **centroides** detalhados de cada cluster.

- **`municipios_por_cluster.xlsx`**  
  - Uma aba por cluster, listando municípios e principais indicadores, **ordenados por IFDM**.

---

## Autoria e Citação

**Autora:** Beatriz Santos  
**Contato:** beatrizfsantos@usp.br

**Citação sugerida:** (sujeita a alterações)

> SANTOS, B. F. S.; THEODORO, R. *Segmentação Socioeconômica de Municípios para Alocação Eficiente de Recursos*. Trabalho de Conclusão de Curso (MBA em Data Science & Analytics) — USP/ESALQ, 2025.

---

## Licença

Este projeto é distribuído sob a **licença MIT**. Consulte o arquivo `LICENSE` para detalhes.