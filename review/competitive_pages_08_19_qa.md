# QA visual: páginas 8–19 da versão competitiva

Data: 2026-09-08. As doze páginas foram abertas individualmente com `view_image`, duas por chamada, nas imagens de 1.650 pixels fornecidas em `tmp/competitive-paper/render`. A revisão se refere a esta renderização; alterações posteriores exigem comparar hashes e rever as páginas afetadas. Nenhum arquivo TeX foi alterado nesta tarefa.

**Resultado: nenhum defeito visual ou numérico bloqueante encontrado.** Não há cortes nas figuras, sobreposição de texto, cabeçalhos isolados sem texto de apoio ou referências indefinidas visíveis nas páginas inspecionadas.

| Página | Elementos examinados | Resultado |
| --- | --- | --- |
| 8 | Figura 4; controles espaciais; início da Seção 5 | Ranking exploratório identificado; números dos controles e p ajustado corretos; conteúdo dentro das margens. |
| 9 | Figura 5; modelo LPQ e CNN original | Quatro condições sobre as mesmas geometrias; legenda explica preservação e seleção dos exemplos. Texto interno da figura é pequeno, porém legível na renderização. |
| 10 | Figura 6; resultados da CNN original; início da fusão | Contrastes e intervalos dos controles não cortados; p de Holm 0,185 arredonda corretamente 0,1848. |
| 11 | Figura 7; Equação 8; início da comparação competitiva | Legendas, eixos, símbolos e introdução de seção legíveis. A comparação anterior permanece condicionada a SAZ. |
| 12 | Figura 8; Tabela 3; protocolo competitivo | Fusão antiga identificada; 83 atributos discriminados em 11+24+36+12; tabela legível sem colisão com o texto. |
| 13 | Cinemática, transferência, resultados T234 e all8, início do XAI | Valores 0,6449/0,6143 e diferença 0,0307 conferem; transferência 0,5813 contra 0,5618; separação entre fusão fixa e escolhida internamente explícita. |
| 14 | Tabela 4; Figura 9; método XAI | As quatro linhas conferem com CSV: F1 de fusão 0,6143/0,6449/0,6279/0,6021 e acurácias 61,60/64,80/62,93/60,27%. Intervalos corretos; figuras sem cortes. |
| 15 | Figura 10; XAI; início da Discussão | Quatro contrastes do subconjunto e seus sinais corretos; todos os intervalos cruzam zero e p ajustados são 1. Introdução de seção presente. |
| 16 | Figura 11; discussão dos novos controles | Casos corretos e incorretos legíveis; mapa Grad-CAM zero identificado. Estatísticas da seleção interna e dos controles corretas. |
| 17 | Figura 12; discussão competitiva e literatura | Figuras e legendas claras; resultados preservam incerteza. Acurácias individuais LPQ T2/T3/T4 54,67/56,00/63,20% conferem. |
| 18 | Tabela 5; limitações; reprodutibilidade; conclusão; referências | Comparadores, tarefas e quatro novas acurácias corretos. Quebras de linha da tabela não ocultam valores. Referências começam em ordem alfabética. |
| 19 | Continuação das referências | Entradas legíveis, sem referências indefinidas ou cortes; ordem alfabética preservada. Espaço restante ao fim da bibliografia é normal. |

## Ajustes editoriais opcionais comunicados à raiz

1. A Tabela 4 usa `Accuracy (%)`; explicitar que essa acurácia é da **fusão** evitaria confusão com a média das tarefas, também presente na tabela.
2. Algumas citações repetem os autores: p. 10, “Diaz et al. (Diaz et al., 2019)”; p. 12, “Drotár et al. (Drotár et al., 2016)”; p. 17, “Casademunt González (Casademunt González, 2023)”. Ajustar a forma narrativa da citação melhoraria a leitura.
3. As figuras novas de vários painéis têm rótulos menores que o texto principal. São legíveis nesta versão; ampliar tipografia pode ser considerado durante a futura redução do artigo, sem alterar sua interpretação.

Não foram encontradas inconsistências de ordem entre as Figuras 4–12 ou entre as Tabelas 3–5. A bibliografia autor-data não exige a ordem de primeira aparição das citações. Esta QA não substitui a auditoria dos modelos e métricas, concluída separadamente.

## Hashes da versão inspecionada

| Arquivo | SHA-256 |
| --- | --- |
| page-08.png | 770ec1f62c763873003c1615f5c5e8f37d41d1eb3e30d2d9d0648027cb888c4d |
| page-09.png | d2eb9dc0a7161a0bad0b0b648cecd7580cba916963a9ec9b0fbc0769612f98b5 |
| page-10.png | d11d18060bd1341580344f12cadf829bf79c3aa904f92022436eb40d2bd60144 |
| page-11.png | a7fe6c697e75ff0b7bdb2c8920e29cb3c23d9996d157347440ef9a31adfd58f8 |
| page-12.png | 0cfe0b818d55d2abbbeeda3591fa584d963a42068c7bac85c62f2c33d8977550 |
| page-13.png | 2382ffd85c5683242ef3ef190b4797d11990dbc2dfe0ac4f2052ca3b339ea39e |
| page-14.png | b1cea0a627e4116f640bbe1a05c55e765c5f88402ca1f88d64d03b236631597b |
| page-15.png | 60f3a5ba8a8d6fe09cb0fc6850d5172690da93381a1855b8a883e83702c70551 |
| page-16.png | df3a2b3ff5689ee3fb872ec6a7bae08adce309d14f6ae312d85901b2596b6449 |
| page-17.png | e668f2f70b956624574e2619ff97660e3e5dd03efb2a7b848242d99d23095a11 |
| page-18.png | ce8ac86a2f779f8351748b14a4c274652d1cc71f0f964791cbdd505934fe91d5 |
| page-19.png | 77fc3ed839a2c315799a19f3e626df12f199fabe09c6a30ea106eb3ac739fcc3 |
| manuscript.pdf | 2771879230b42df708e960e387246ba89e362fe156e70a63addbe4eb618309d5 |
| manuscript_with_figure_alternatives.pdf | 70814ad4b7f36e1dbd63edd73c5546e8dde844464cd28cacb184ed30dd0aee62 |

## Rodada final após correções

Revisão visual concluída em 2026-09-08 nos PNGs de `tmp/competitive-paper/render-final`. As páginas 8, 9, 10, 11, 12, 14, 16 e 17 foram reabertas e inspecionadas individualmente. A comparação SHA-256 confirmou que as páginas 13, 15, 18 e 19 permanecem idênticas às imagens já inspecionadas na primeira rodada.

**Resultado: nenhum bloqueante visual; nenhum corte de texto, tabela, figura ou legenda; nenhuma sobreposição após a recomposição local.**

- A Tabela 4, p. 14, agora explicita `Fusion acc. (%)`, com cabeçalho e valores dentro dos limites da tabela. A ambiguidade apontada no item 1 foi resolvida.
- As citações narrativas de Diaz, Drotár e Casademunt foram corrigidas nas páginas 10, 12 e 17. O item 2 foi resolvido.
- As páginas 8–9 apresentam a análise histórica de SAZ com a ressalva de seleção pelos resultados externos, sem problemas de fluxo ou hierarquia das subseções.
- Na p. 16, o texto esclarece que a comparação entre procedimentos adaptativos não isola o efeito da escolha da regra de fusão. Na p. 17, a redução para três tarefas não é apresentada como evidência de menor tempo de avaliação ou carga clínica medida.
- As quebras de palavras entre as páginas 8–9 e 11–12 correspondem à hifenização normal do texto; não há conteúdo perdido. As métricas principais, figuras e referências mantêm a coerência registrada na primeira rodada.

A observação opcional sobre o tamanho da tipografia interna das figuras permanece apenas como sugestão para a futura redução do artigo. Não exige alteração para esta versão de revisão dos orientadores. Nenhum arquivo TeX foi editado nesta QA.

### Hashes finais

| Arquivo | SHA-256 |
| --- | --- |
| page-08.png | 01fd5c78fc70e92a0d97a97389557d4ad29630528cbd13d912f6a87cc352ba80 |
| page-09.png | d34853d5af48dfbe897378a3ccef766d1a6655080572eab54b4b35321030a2d1 |
| page-10.png | 30c873e40431bdacc22253ac413249bd79ee687df0e8e43c72aa0aa64d4d84ec |
| page-11.png | 0c8bf9180d28d82d83b93c81f912765e9f1b0ba9738a6c3089f1c5c4cc90042e |
| page-12.png | 901b8c1c0c1a71b6b20f44f4a0c1f2c4d2199870a13361834607a824c6c1510f |
| page-13.png | 2382ffd85c5683242ef3ef190b4797d11990dbc2dfe0ac4f2052ca3b339ea39e |
| page-14.png | f7c02f3382edc27a8ea11807b0e5cf3dd3c63840742865e5eddd2970fba77057 |
| page-15.png | 60f3a5ba8a8d6fe09cb0fc6850d5172690da93381a1855b8a883e83702c70551 |
| page-16.png | cdcaf6d1753c55a77c5b5b794074315fb49a4dd0df573a94e0022fa65538289e |
| page-17.png | b339108c434fba37dd622e4fbef56fdeec85d17e28baef6fc5617a086160f628 |
| page-18.png | ce8ac86a2f779f8351748b14a4c274652d1cc71f0f964791cbdd505934fe91d5 |
| page-19.png | 77fc3ed839a2c315799a19f3e626df12f199fabe09c6a30ea106eb3ac739fcc3 |
| manuscript.pdf | ebdf892823be26670dab043d3a28817a935308f339560c130e1f98b82fc33775 |
| manuscript_with_figure_alternatives.pdf | 657a121c4a6af85efe3040bdb103cf9137f619b5e9a832dd047a453a75c82eba |
