# Rossman Drugstore Sales Prediction

![Rossman!](img/rossmann_logo.png)
# Introdução 
Esse é um projeto end-to-end de Data Science com modelo de regressão adaptada para séries temporais. No qual criamos 4 tipos de modelos para predizer o valor das vendas das lojas nas próximas 6 semanas. As previsões podem ser acessadas pelo usuário por meio de um BOT no aplicativo do Telegram.

Este repositório contém a solução para a resolução de uma problema do Kaggle: https://www.kaggle.com/c/rossmann-store-sales 

Esse projeto faz parte da "Comunidade DS", que é um ambiente de estudo que promove o aprendizado, execução, e discussão de projetos de Data Science.

### Plano de Desenvolvimento do Projeto de Data Science
Esse projeto foi desenvolvido seguindo o método CRISP-DS(Cross-Industry Standard Process - Data Science). Essa é uma metodologia capaz de transformar os dados da empresa em conhecimento e informações que auxiliam na tomada de decisão. A metodologia CRISP-DM define o ciclo de vida do projeto, dividindo-as nas seguintes etapas:
* Entendimento do Problema de Negócio 
* Coleção dos Dados
* Limpeza de Dados
* Análise Exploratória dos Dados
* Preparação dos Dados
* Modelos de Machine Learning, Cross-Validation e Fine-Tuning.
* Avaliação dos Resultados do Modelo e Tradução para Negócio.
* Modelo em Produção

![crisp!](img/crisp.png)

### Planejamento
* [1. Descrição e Problema de Negócio](#1-descrição-e-problema-de-negócio)
* [2. Base de Dados e Premissas de Negócio](#2-base-de-dados-e-premissas-de-negócio)
* [3. Estratégia de Solução](#3-estratégia-de-solução)
* [4. Exploration Data Analysis](#4-exploration-data-analysis)
* [5. Seleção do Modelo de Machine Learning](#5-seleção-do-modelo-de-machine-learning)
* [6. Performance do Modelo](#6-performance-do-modelo)
* [7. Resultados de Negócio](#7-resultados-de-negócio)
* [8. Modelo em Produção](#8-modelo-em-produção)
* [9. Conclusão](#9-conclusão)
* [10. Aprendizados e Trabalhos Futuros](#10-aprendizados-e-trabalhos-futuros)



# 1. Descrição e Problema de Negócio

### 1.1 Descrição
**Rossman Sales Drugstore** é uma empresa que opera mais de 3000 drogarias em 7 países europeus. Atualmente os gerentes da loja Rossman têm a tarefa de prever suas vendas diárias com até seis semanas de antecedência. As vendas das lojas são influenciadas por muitos fatores, incluindo promoções, competição, feriados escolares e estaduais, sazonalidade e localidade. Atualmente essa previsão é feita por meio de uma simples média das vendas de cada loja. 

### 1.2 Problema de Negócio
Foi feita uma reunião entre o CEO e os sócios da Rossman e foi definido que a empresa irá investir nas reformas das lojas da rede Rossman. Para que essa reforma seja possível será necessário prever o valor de vendas de cada loja de maneira mais assertiva, para que assim o CEO tenha uma melhor noção do quanto investir em cada loja. 

Dito isso,a empresa decidiu contratar um Cientista de Dados para realizar as seguintes tarefas:

**- Realizar a previsão das vendas de cada uma das lojas nas pŕoxima seis semanas.**

**- Fornecer ao CEO uma forma de consulta rápida dessas previsões por meio do celular.**


# 2. Base de Dados e Premissas de Negócio
## 2.1 Base de Dados
O conjunto de dados total possui as informações referentes 1115 lojas e possuem os seguintes atributos:
| **Atributos** |  **Descrição**  |
| ------------------- | ------------------- |
|  id | Um Id que representa um (Store, Date) concatenado dentro do conjunto de teste |
|  Store |  Um id único para cada loja |
|  Sales |  O volume de vendas em um determinado dia |
|  Customers |  O número de clientes em um determinado dia |
|  Open |  Um indicador para saber se a loja estava aberta: 0 = fechada, 1 = aberta |
|  StateHoliday |  Indica um feriado estadual. Normalmente todas as lojas, com poucas exceções, fecham nos feriados estaduais. Observe que todas as escolas fecham nos feriados e finais de semana. a = feriado, b = feriado da Páscoa, c = Natal, 0 = Nenhum |
| SchoolHoliday |  Indica se (Store, Date) foi afetada pelo fechamento de escolas públicas |
|  StoreType |  Diferencia entre 4 modelos de loja diferentes: a, b, c, d |
|  Assortment |  Descreve um nível de sortimento: a = básico, b = extra, c = estendido |
|  CompetitionDistance |  Distância em metros até a loja concorrente mais próxima |
|  CompetitionOpenSince[Month/Year] |  Apresenta o ano e mês aproximados em que o concorrente mais próximo foi aberto |
|  Promo |  Indica se uma loja está fazendo uma promoção naquele dia |
|  Promo2 |  Promo2 é uma promoção contínua e consecutiva para algumas lojas: 0 = a loja não está participando, 1 = a loja está participando |
|  Promo2Since[Year/Week] |  Descreve o ano e a semana em que a loja começou a participar da Promo2 |
|  PromoInterval | Descreve os intervalos consecutivos de início da promoção 2, nomeando os meses em que a promoção é iniciada novamente. Por exemplo. "Fev, maio, agosto, novembro" significa que cada rodada começa em fevereiro, maio, agosto, novembro de qualquer ano para aquela loja |
## 2.2 Premissas de Negócio
Para realizar esse projeto as seguintes premissas de negócio foram adotadas:
* Os dados de costumers foram descartados, visto que para utilizar esse atributo teríamos que calcular uma previsão de número de clientes que pode-se tornar um projeto a parte complementar a este.
* Os dias que as lojas encontram-se fechadas foram descartadas.
* Só foram consideradas as entradas que obtiveram o valor de venda ("SALES") maior que 0.
* Para lojas que não tinham informação de Competition Distance foi adotado um valor arbitrário alto para efeitos de comparação.
# 3. Estratégia de Solução
A estratégia de solução foi a seguinte:
### Passo 01. Descrição dos Dados
Nesse passo foi verificado alguns aspectos do conjunto de dados, como: nome de colunas, dimensões, tipos de dados, checagem e preenchimento de dados faltantes (NA), análise descritiva dos dados e quais suas variáveis categóricas.
### Passo 02. Featuring Engineering
Na featuring engineering foi derivado novos atributos(colunas) baseados nas variáveis originais, possibilitando uma melhor descrição do fenômeno daquela variável.

### Passo 03. Filtragem de Variáveis
O conjunto de dados foi filtrado por linhas para que levássemos em consideração apenas as lojas que estão abertas e que realizaram vendas ( open != 0 e sales > 0) e por coluna foi feita um drop das variáveis que não agregam valor de conhecimento ou foram derivados para outras variáveis.
### Passo 04. Análise Exploratória dos Dados (EDA)
Exploração dos Dados com objetivo de encontrar Insights para o melhor entendimento do Negócio. 
Foram feitas também análises univariadas, bivariadas e multivariadas, obtendo algumas propriedades estatísticas que as descrevem, e mais importante  a correlação entre as variáveis.
### Passo 05. Preparação dos Dados
Sessão que trata da preparação dos dados para que os algoritmos de Machine Learning possam ser aplicados. Foram realizados alguns tipos de escala e encoding para que as variáveis categóricas se tornassem numéricas.
### Passo 06. Seleção de Variáveis do Algoritmo
A seleção dos atributos foi realizada utilizando o método de seleção de variáveis Boruta. No qual os atributos mais significativos foram selecionados para que a performance do modelo fosse maximizada.
### Passo 07. Modelo de Machine Learning
Realização do treinamento dos modelos de Machine Learning . O modelo que apresentou a melhor perfomance diante a base de dados com cross-validation aplicada seguiu adiante para a hiper parametrização das variáveis daquele modelo, visando otimizar a generalização do modelo.
### Passo 08. Hyper Parameter Fine Tuning
Foi encontrado os melhores parâmetros que maximizavam o aprendizado do modelo. Esses parâmetros foram definidos com base no método de RandomSearch.
### Passo 09. Conversão do Desempenho do Modelo em Valor de Negócio
Nesse passo o desempenho do modelo foi analisado mediante uma perspectiva de negócio,e traduzido para valores de negócio.
### Passo 10. Deploy do Modelo em Produção 
Publicação do modelo em um ambiente de produção em nuvem (Heroku) para que fosse possível o acesso de pessoas ou serviços para consulta dos resultados e com isso melhorar a decisão de negócio da empresa.

### Passo 11. Telegram Bot
Criação de um bot no Aplicativo de mensagens do Telegram. Cuja consulta das previsões podem ser feitas de qualquer lugar a qualquer momento apenas utilizando uma conexão com a internet e o aplicativo no smartphone.

# 4. Exploration Data Analysis 
## 4.1 Análise Univariada
* Variáveis Numéricas: o histograma abaixo mostra como está organizada a distribuição das variáveis numéricas do nosso conjunto de dados. Mostra a contagem de cada variável numérica do dataset.

![Numerical-Variables!](img/analise_univariada.png)

## 4.2 Análise Bivariada
### H2. Lojas com competidores mais próximos deveriam vender menos.
**FALSO** Lojas com competidores MAIS próximos vendem MAIS.
* No 1º gráfico podemos ver que a maioria dos dados estão concentrados num range de distância de 0 a 25000. 
* No 2º gráfico foi feito um agrupamento por intervalos de distância, como observado as lojas que tem competidores mais próximos tem mais vendas.
* O heatmap demonstra uma correlação negativa, isso significa que a variável tem uma relevância média na influência das vendas e quando há influência, é no geral negativa.
 
![H2!](img/h2.png)

### H3. Lojas com competidores à mais tempo deveriam vender mais.
**FALSO**  Lojas com competidores à MAIS tempo vendem MENOS. 
* A variável "competition_time_month" foi criada e indica há quanto tempo em meses aquela loja enfreta uma competição. PS: valores negativos significam que o competidor ainda não iniciou as vendas.
* Podemos ver que as lojas que têm competidores há mais tempo vendem menos, devido ao segundo gráfico, que mostra um comportamento de decaímento de vendas com o aumento do tempo de competição.
* Através do heatmap vemos uma correlação fraco para o modelo.

![H3!](img/h3.png)

### H9. Lojas deveriam vender mais no segundo semestre do ano
**FALSO**  Lojas vendem MENOS no segundo semestre do ano.
* Como podemos ver, durante os 6 primeiros meses as lojas vendem mais do que o resto do ano.
* Correlação muito forte negativamente, essa é considerada uma das variáveis mais importantes para o modelo. 

![H9!](img/h9.png)

### H10. Lojas deveriam vender mais depois do dia 10 de cada mês
**VERDADEIRO**  Lojas vendem MAIS após dia 10 de todo mês.
* Nessa hipótese a ideia era verificar se as vendas no início do mês, no qual geralmente é feito o pagamento de salário, conseguiria alcançar as vendas nos 20 dias restantes do mês.
* Lojas vendem menos no período inicial de 10 dias de cada mês.
* Correlação Negativa.

![H10!](img/h10.png)

### H11. Lojas deveriam vender menos aos finais de semana
**VERDADEIRO** Lojas vendem MENOS aos finais de semana.
* Tendência de cair as vendas com o passar dos dias da semana.
* No fim de semana as vendas caem drásticamente, principalmente no domingo.
* Correlação forte negativamente. Isso significa que se as lojas se encontrarem no período de final de semana, irão vender menos.

![H11!](img/h11.png)

### Tabela de Insights 

| Hipóteses | Condição| Relevância |
| :-------- | :------- | :--------  |
|H1. Lojas com maior sortimento deveriam vender mais|Falsa|Baixa|
|H2. Lojas com competidores mais próximos deveriam vender menos.|Falsa|Média|
|H3. Lojas com competidores à mais tempo deveriam vender mais.|Falsa|Média|
|H4. Lojas com promoções mais ativas por mais tempo deveriam vender mais.|Falsa|Baixa|
|H5. Lojas com mais dias de promoção deveriam vender mais.| --- |---|
|H6. Lojas com mais promoções consecutivas deveriam vender mais.|Falsa|Baixa|
|H7. Lojas abertas durante o feriado de Natal deveriam vender mais|Falsa|Média|
|H8. Lojas deveriam vender mais ao longo dos anos.|Falsa|Alta|
|H9. Lojas deveriam vender mais no segundo semestre do ano.|Falsa|Alta|
|H10 .Lojas deveriam vender mais depois do dia 10 de cada mês.|Verdadeira|Alta|
|H11 .Lojas deveriam vender menos aos finais de semana.|Verdadeira|Alta|
|H12 .Lojas deveriam vender menos durante os feriados escolares.|Verdadeira|Baixa|


## 4.3 Análise Multivariada

![multivariate-analysis!](img/correlacao_numerica.png)
 
 ### Correlação entre as variáveis independentes e a variável resposta
 * Variáveis com correlação positiva com sales:
   * **Média:** > *promo*
   * **Fraca:** > *competition_open_since_year, promo2_since_year*

* Variáveis com correlação negativa com as vendas:
  * **Média:** > *day_of_week*
  * **Fraca:** > *promo2, is_promo*

# 5. Seleção do Modelo de Machine Learning 
Os seguintes algoritmos de Machine Learning foram aplicados:
* Mean Average Model (Usado como Baseline);
* Linear Regression Model;
* Linear Regression Regularized Model - Lasso;
* Random Forest Regression;
* XGBoost Forest Regression;

O método de cross-validation foi utilizado em todos os modelos.

# 6. Performance do Modelo
O modelo RandomForestRegressor foi o modelo que apresentou o melhor desempenho em Single Performance, com um percentual de Erro médio (MAPE) de aproximadamente 10%. No entanto nesse projeto foi optado a utilização do modelo **XGBoost Regressor** visto que a performance do modelo é equivalente a RandomForest, e ele lida melhor com base de dados maiores e tende a ser mais rápido em sua etapa de treinamento do modelo.

|Model Name	| MAE	| MAPE	| RMSE|
|---------|----|----|----|
|Random Forest Regressor	|667.434016	|0.097598	| 996.039250
|**XGBoost Regressor**	|**756.970422**	|**0.111236**| **1095.040393**|
|Average Model	|1354.800353	|0.455150	|1835.141019|
|Linear Regression	|1867.269017|	0.292706|	2671.589246|
|Linear Regression Regularized - Lasso	|1891.702083	|0.289165	|2744.462516|

A real performance dos modelos utilizando método CROSS-VALIDATION.

|Model Name	|MAE CV	|MAPE CV	|RMSE CV|
|----------| -------|---------|--------|
|Random Forest Regression	|843.422+/- 223.13	|0.12 +/- 0.02	|1267.99 +/- 328.10|
|**XGBoost Regressor**|**1122.76.71 +/- 212.97**	|**0.15 +/- 0.02**	|**1622.56 +/- 317.51**|
|Linear Regression	|2081.69 +/- 295.46	|0.3 +/- 0.02	|2952.42 +/- 468.15|
|Linear Regression Regularized	|2116.64 +/- 341.57	|0.29 +/- 0.01	|3057.93 +/- 504.72|

Escolhido o modelo de Regressão XGBoost partimos para a etapa de HyperParamater Fine-Tuning que consiste em encontrar os melhores parâmetros de treino para maximizar o aprendizado do modelo, usamos o Random Search e definimos os parâmetros através do melhor resultado encontrado. Após encontrarmos os valores ótimos para o modelo por meio do método RandomSearch os valores finais de desempenho do modelo foram:


## Final-Performance Fine-Tuned CV Model
|Model Name | MAE CV | MAPE CV | RMSE CV |
|-----|----|----|-----
|XGBoost Regressor | CV	687.286713 |0.101122	|990.867971

# 7. Resultados de Negócio
Com base no método atual de previsão de vendas é possível analisarmos a diferença de performance entre o modelo utilizado (Average Model) e o modelo proposto XGBoost Regressor.

**Modelo Atual baseado na média de vendas**

|Cenário| Valores|
|---|---|
| Valor Real das Vendas| R$280.754.389,45| 

**Modelo XGBoost sugerido**<br>
A diferença é com relação ao valor real das vendas e a porcentagem desse valor.
|Cenário|Valores|Diferença|%|
|------|------|------|------|
|Predições|	R$284,615,072.00|R$3.860.682,55|1,3751%|
|Pior Cenário|	R$283,844,721.87|R$3.090.332,42|1,1101%|
|Melhor Cenário|	R$285,385,428.41|R$4.631.038,96|%1,6495|

O modelo XGboost teve um bom desempenho para as lojas Rossman, porém tivemos algumas lojas as quais o MAPE ficou muito acima do normal, como podemos ver nas tabelas e gráficos abaixo:

* Nesse gráfico, podemos observar a distribuição dos erros MAPE para todas as lojas da rede Rossman. É importante notar que existem algumas lojas em específico que são mais desafiadoras que outras na previsão das vendas. Porém, é razoável assumir que nosso modelo desempenhou bem de maneira geral visto que a maioria das nossas lojas está concentrada em uma área com um MAPE próximo a 10%. É claro que, o CEO e a equipe de negócio devem analisar esses dados e definir se é "aceitável" esse valor de erro ou não. 
![error!](img/store_x_mape.png)

* Alguns outros gráficos foram plotados com objetivo de fornecer um maior entendimento ao time de negócio como o nosso modelo se comporta de maneira geral. O 1º gráfico nos mostra os valores de vendas (linhas azuis) e a predição do modelo (linhas laranjas) das últimas 6 semanas de venda. Como podemos ver, nosso modelo se comporta muito bem, acompanhando de perto as vendas reais.

* No 2º gráfico é representado a taxa de erros com relação às vendas. Essa taxa é calculada pela razão entre os valores preditos e os valores reais de venda observados. Visto que o modelo não possui taxas muito exorbitantes, podemos assumir que ele teve um bom desempenho.

* No 3º gráfico podemos ver a distribuição da taxa de erro, que tem como característica uma forma normal cujo centro está tendendo a 0.

* O 4º gráfico demonstra a dispersão que representa as previsões realizadas em relação aos erros de cada dia de venda. Idealmente, nós teríamos nossos pontos concentrados e formaríamos uma espécie de "tubo", pois dessa forma ela representaria uma baixa variação de erro em todos valores que a previsão de vendas poderia assumir.
![erro2](img/ml_performance.png)


# 8. Modelo em Produção
O modelo de Machine Learning foi implementado e colocado em produção por meio da plataforma Render (https://render.com), que tem como objetivo possibilitar a criação, execução e operação de aplicativos inteiramente localizados em nuvem. 

## Esquemático do deploy do modelo em produção

![telegram_modelo](img/telegram_modelo.png)

Além do modelo em si presente na nuvem, foi criado um BOT no aplicativo do Telegram que possibilita ao CEO e os time de negócio da empresa realizarem consultas da previsão de vendas das lojas nas próximas 6 semanas de forma simples e direta. Basta apenas utilizar um smartphone e enviar uma mensagem ao bot no Telegram localizado no endereço: http://t.me/projeto_rossmann_bot.

Forma de Utilização:
* Criar conta no Telegram em seu smartphone e abrir o link citado acima.
* Enviar o número de loja que deseja saber a previsão de venda.

![telegram!](img/telegram.jpeg)

# 9. Conclusão
Nesse projeto, foram realizadas todas as etapas necessárias para a implementação de um projeto completo de Data Science em um ambiente de produção. Foi utilizado o método de gerenciamento de projeto chamado CRISP-DM/DS e obteve-se um desempenho satisfatório utilizando o modelo de Regressão XGBoost para realizar a previsão de venda das lojas da rede Rossman para as próximas 6 semanas.

Vários Insights de Negócio foram gerados durante a Análise Exploratória de dados que ajudaram o CEO, junto ao time de negócio e o cientista de dados a entenderem melhor o negócio. Tendo em vista esses resultados, o projeto alcançou seu objetivo de encontrar uma solução simples e assertiva para previsão de vendas das lojas, disponibilizando um BOT no Telegram que retorna as previsões geradas pelo modelo de forma rápida e eficaz.

# 10. Aprendizados e Trabalhos Futuros

**Aprendizados**

* Esse problema de predição foi resolvido utilizando técnicas de Regressão adaptadas a Time-Series.

* A escolha do modelo de Machine Leaning deve-se levar em consideração não só a performance do modelo em si, mas também sua generabilidade, levando em consideração o custo de implementação, a dificuldade e tempo de execução de todo projeto.

* A Análise Exploratória de Dados se demonstrou uma das etapas mais importantes do projeto, pois é nessa parte que podemos encontrar Insights de Negócio que promovem novos conhecimentos e até contradições que nos fazem repensar o negócio como um tudo. Essa análise também fornece ao cientista de dados uma "direção" de como melhorar seu modelo, por meio da criação de novas features e diferentes tipos de abordagem.

**Trabalhos Futuros**
* Entender melhor as lojas com taxas de erros muito elevadas e trabalhar talvez e um modelo específico para elas.
* Criar novas features a partir de variáveis com correlação forte com a variável resposta.
* Criar um modelo específico para analisarmos o número de clientes de cada loja.
* Tentar métodos de Encoding diferentes para melhor performance do modelo.
* Fazer outro Fine Tuning, afim de encontrar um um erro menor e com isso melhorar a performance do modelo.
* Apresentar mais opções de mensagens no Telegram, podendo adicionar mais de uma loja por mensagem, com mais condicionais.





