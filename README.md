# Previsão de demanda no varejo: M5 Forecasting

Previsão das vendas diárias dos próximos 28 dias para 30.490 séries (item × loja) do Walmart, com os dados da competição [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) do Kaggle. O notebook cobre análise exploratória, definição de métrica e validação, baselines, ETS, SARIMA e LightGBM, todos comparados no mesmo holdout.

## Resultado

Erro no holdout de 28 dias (25/04 a 22/05/2016), medido em WRMSSE (menor é melhor):

| Modelo | WRMSSE |
|---|---|
| LightGBM com lags | 0,619 |
| SARIMA (top-down por loja) | 0,725 |
| Média sazonal de 4 semanas | 0,752 |
| ETS (top-down por loja) | 0,781 |
| Naive sazonal (7 dias) | 0,870 |
| Média móvel 28 dias | 1,082 |
| Média móvel 7 dias | 1,136 |
| Naive | 1,464 |

O LightGBM reduz o erro em 17,7% em relação ao melhor baseline (média sazonal). Nos níveis mais desagregados da hierarquia (item e item-loja) a diferença entre os três melhores modelos é pequena, e a vantagem do LightGBM se concentra nos níveis agregados.

![Comparação dos modelos](reports/figures/comparacao_modelos.png)

## Dados

Três tabelas, com 1.941 dias de vendas, 3 estados, 10 lojas e 3.049 itens:

- `sales_train_evaluation.csv`: vendas diárias por item e loja, em formato wide (uma coluna por dia).
- `calendar.csv`: datas, dia da semana, eventos e dias de SNAP.
- `sell_prices.csv`: preço semanal por item e loja.

Os arquivos não estão no repositório. O notebook lê de `data/m5-forecasting-accuracy/`.

## Metodologia

**Validação.** Holdout nos últimos 28 dias da base (`d_1914` a `d_1941`). Os baselines também passam por walk-forward em 3 origens (`d_1857`, `d_1885`, `d_1913`) para checar se o ranking se mantém. Os demais modelos são avaliados só no fold final.

**Métrica.** WRMSSE com os 12 níveis da hierarquia do M5. O erro de cada série é dividido pelo erro do naive de 1 passo no treino e ponderado pela receita dos últimos 28 dias. A implementação é própria e não foi conferida contra o avaliador oficial, então os valores comparam os modelos entre si, não com o leaderboard da competição.

**Modelos.**

- Baselines no nível item-loja: naive, naive sazonal, médias móveis de 7 e 28 dias e média sazonal de 4 semanas.
- ETS (Holt-Winters amortecido) e SARIMA (1,1,1)(0,1,1,7), ajustados na série total de cada loja e distribuídos entre os itens por proporção histórica (top-down).
- LightGBM com objetivo Tweedie, um modelo por loja e previsão direta de 28 dias. As features de histórico só usam dados até 28 dias antes do alvo (lags, médias móveis, desvio padrão, fração de zeros), junto com preço, calendário, eventos e SNAP.

## Estrutura

```
.
├── eda_forecasting_m5.ipynb      # EDA, validação, modelos e comparação
├── reports/figures/              # gráficos exportados pelo notebook
├── data/m5-forecasting-accuracy/ # CSVs do Kaggle (não versionado)
├── requirements.txt
└── README.md
```

## Como reproduzir

```bash
git clone https://github.com/GustavoSousa777/<nome-do-repositorio>.git
cd <nome-do-repositorio>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Baixe os dados em [kaggle.com/competitions/m5-forecasting-accuracy/data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) (é preciso aceitar as regras da competição) e extraia `calendar.csv`, `sales_train_evaluation.csv` e `sell_prices.csv` em `data/m5-forecasting-accuracy/`. Com a CLI do Kaggle:

```bash
pip install kaggle
kaggle competitions download -c m5-forecasting-accuracy -p data/m5-forecasting-accuracy
unzip data/m5-forecasting-accuracy/m5-forecasting-accuracy.zip -d data/m5-forecasting-accuracy
```

Depois, abra `eda_forecasting_m5.ipynb` e rode as células em ordem.

Testado com Python 3.12, pandas 3.0, numpy 2.5, statsmodels 0.15 e LightGBM 4.7. As seções de modelagem (4 a 8) levam cerca de 10 minutos em um único núcleo. A EDA da seção 3 é a parte mais pesada em memória, então recomendo 8 GB de RAM ou mais.

## Limitações

- Um único holdout de 28 dias, sem Natal. Só os baselines passaram por walk-forward.
- LightGBM quase sem tuning: o número de árvores vem de early stopping em uma única loja. As ordens do SARIMA são fixas.
- ETS e SARIMA só no nível de loja, com proporções fixas entre itens e sem reconciliação hierárquica.
- Previsão pontual, sem intervalos de incerteza.

## Próximos passos

- Walk-forward também para o LightGBM.
- Tuning com Optuna e rastreamento de experimentos com MLflow.
- Features de evento mais ricas (dias até e desde o evento).
- Reconciliação hierárquica das previsões.
- Comparação com deep learning e foundation models para séries temporais.

## Autor

Gustavo Pereira Sousa · [LinkedIn](https://linkedin.com/in/gustavosousa777) · [GitHub](https://github.com/GustavoSousa777)
