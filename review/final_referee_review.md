# Revisão final como parecerista — 8 de setembro de 2026

Este é um parecer simulado para revisão dos autores, sem representar uma decisão
do evento. A avaliação usa os critérios oficiais do
[VISAPP 2027](https://visapp.scitevents.org/Guidelines.aspx?y=2027): relevância,
originalidade, qualidade técnica, significância e apresentação. As práticas de
três artigos publicados no evento, examinadas em `bibliography_audit.md`, servem
como referências de apresentação e de delimitação das conclusões.

## Avaliação científica

O argumento mais forte é a formulação explícita da representação e sua avaliação
incremental com partições comuns por participante. O estudo tem pertinência para
análise de imagens, cor, textura e explicação de classificadores. A contribuição
deve ser julgada como investigação metodológica exploratória: os resultados não
sustentam superioridade geral sobre a imagem estática, estado da arte em PaHaW
ou validação clínica. Os principais riscos de avaliação continuam sendo a
significância limitada do ganho global e a necessidade de demonstrar claramente
o que a formulação acrescenta aos antecedentes de dinâmica representada em imagens.

## Fragilidades corrigidas nesta versão

| Problema de revisão | Correção aplicada |
|---|---|
| A conclusão enfatizava a ausência de significância e ocultava os ganhos observados que a abertura já descrevia. | Discussão e conclusão agora registram quatro tarefas com ganho, o aumento na espiral e a diferença média, preservando a incerteza da comparação global. |
| A novidade poderia ser confundida com a invenção de toda codificação visual de dinâmica. | A introdução identifica a combinação específica de trajetória em contato, normalização por registro e centralização circular do azimute, mantendo os antecedentes de Diaz e Cilia. |
| A descrição das famílias de testes omitia as comparações entre regras de fusão. | Incluídas as três comparações de fusão SVM; explicitado que os intervalos são percentis do bootstrap estratificado por diagnóstico. |
| O tratamento dos mapas Grad-CAM nulos na deleção não estava explícito. | Informado que os 174 mapas nulos permanecem nas 2.985 predições da deleção e que, nesses casos, a escolha dos pixels depende do desempate da rotina. Os números originais foram preservados. |
| A discordância Grad-CAM–oclusão permitia uma interpretação excessiva. | A conclusão limita esse resultado à consistência da interpretação espacial da CNN testada; não o apresenta como prova da ausência de sinais de doença ou explicação de seu desempenho. |
| O texto repetia ressalvas e comentários genéricos sobre a própria análise. | Reduzidas repetições na introdução, resultados e discussão; substituídas frases abstratas por descrição de sinais, métricas, decisões e limites específicos. Mantidos os parágrafos de ligação entre seções e subseções. |
| A reprodutibilidade dependia de código que não acompanhava o artigo. | Incluídos 37 arquivos originais de código, configurações e ambiente para os oito estágios, com README, ordem de execução e manifesto SHA-256. A seção distingue esse pacote do arquivo interno que conserva predições e checkpoints. |

As verificações científicas confrontaram o texto com código e resultados salvos,
incluindo `task-fusion-v1/metrics/global_pairwise_comparisons.csv` e a rotina de
deleção em `src/pdhms_restart/xai.py`. Não houve ajuste de modelos, novo cálculo
de métricas ou seleção adicional de resultados. A bibliografia e as figuras
científicas não foram alteradas nesta rodada.

## Escrita e assistência por IA

A revisão tratou problemas observáveis: repetição de ressalvas, frases genéricas,
contrastes retóricos desnecessários, comentários sobre o próprio texto e falta
de conexão entre descrição e resultado. Foram preservadas a voz científica,
a terminologia técnica e as qualificações necessárias. Esses aspectos não
demonstram autoria por IA, e nenhum detector foi usado. A assistência efetivamente
prestada continua registrada em `ai_disclosure.md`, à luz da
[política do evento](https://visapp.scitevents.org/AiTools.aspx?y=2027).

## Limitações que uma revisão textual não resolve

- A seleção de SAZ usa resultados externos do mesmo conjunto de 75 participantes;
  as comparações posteriores permanecem exploratórias e condicionais a essa escolha.
- O aumento médio de macro F1 de 0.5643 para 0.5752 coexiste com perdas em quatro
  tarefas e não estabelece vantagem global após a análise pareada e as correções.
- A comparação usa uma CNN pequena treinada do zero, com resolução e procedimento
  de seleção diferentes dos modelos LPQ; não autoriza uma conclusão sobre CNNs em geral.
- Os controles de deleção não igualam área de escrita ou forma das regiões removidas.
  As perturbações e a baixa concordância entre mapas limitam a interpretação do XAI.
- As comparações com Casademunt, Drotár e Diaz são contextuais: métricas, tarefas,
  seleção e partições diferem. Nosso desempenho combinado é inferior aos valores
  publicados de Drotár e Diaz, e este estudo não decompõe as causas dessa diferença.

Dados independentes e seleção de representação restrita ao treinamento seriam
necessários para enfrentar essas limitações. Não foram acrescentados experimentos
para tentar resolvê-las por extensão do escopo desta revisão.

## Verificação da entrega

O build final mantém 14 páginas, oito figuras principais, três tabelas e 13
referências citadas; o resumo tem 184 palavras. A cópia de seleção acrescenta
as 12 alternativas nas páginas 15–26. O rascunho ampliado segue a autorização
dos autores para discutir os cortes com Yandre e Ángel.

As páginas 1–3 foram inspecionadas individualmente; o diagrama aprovado permanece
intacto. Os relatórios `final_referee_methods_qa.md` e `final_referee_style_qa.md`
registram a inspeção das páginas 4–14. As páginas 15–26 são idênticas, por hash
dos PNGs renderizados, à galeria individualmente inspecionada na revisão anterior.
O log LaTeX não apresenta referências indefinidas, caracteres ausentes ou caixas
transbordando. `build_verification.json` registra contagens e hashes finais.

A última correção, que identifica o arquivo interno de resultados, alterou apenas
a página 13. Ela foi renderizada e reinspecionada; as outras 25 páginas mantiveram
os mesmos pixels da versão revisada. Os dois pareceres de QA estão encerrados
sem pendências. O texto extraível final contém 40.119 caracteres sem espaços.

O pacote experimental foi auditado quanto a sintaxe, dependências locais,
configurações e integridade, sem execução dos modelos. Ele não redistribui os
registros PaHaW, predições individuais ou checkpoints; o README documenta os
ajustes de caminhos e os limites da regeneração dos gráficos históricos.
