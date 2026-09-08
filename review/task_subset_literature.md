# Subconjunto de tarefas: evidência bibliográfica e decisão recomendada

Verificação em 2026-09-08. Leitura das referências existentes, fontes primárias e tabelas renderizadas; nenhum experimento executado neste levantamento e nenhuma alteração do artigo. As páginas abaixo são posições no PDF, salvo indicação contrária.

## Recomendação para congelar antes da nova execução

Usar **T2, T3 e T4** como conjunto principal da comparação de métodos: repetição da letra cursiva **l**, do bigrama **le** e do trigrama **les**. Esse conjunto corresponde às tarefas reportadas no TFG de Casademunt González, orientado por Ángel Sánchez Calle, explicitamente indicado pelo usuário como comparação prioritária. Há uma progressão de complexidade ortográfica e uma aquisição mais homogênea que a mistura de desenho, palavras e sentença.

A justificativa é a correspondência com a literatura e o escopo da comparação, **não uma superioridade demonstrada dessas três tarefas**. O TFG desenvolve o método principalmente em T2 e apresenta T3/T4 no apêndice; não publica uma fusão conjunta dessas três tarefas. A decisão proposta é nossa extensão experimental e deve ser identificada assim.

Congelar o conjunto para todos os métodos antes de executar a etapa 3. Manter os mesmos participantes e partições externas, escolher os hiperparâmetros somente no treinamento e comparar os métodos dentro de T2–T4. Conservar os resultados anteriores de T1–T8 como avaliação abrangente. A média sobre três tarefas responde a uma pergunta diferente da média sobre oito: seu aumento não demonstra, por si só, melhora do classificador. A coorte já participou do desenvolvimento; fixar agora o conjunto não cria uma validação externa independente.

## O que as fontes efetivamente reportam

| Fonte existente | Tarefas e escolha | Evidência e protocolo |
|---|---|---|
| **Casademunt González, 2023** | Desenvolvimento em T2; extensão a T3/T4. | TFG: resumo; Anexo III, p. 75/impresso 61; Tabela III.6, p. 79/impresso 65. Recortes 111; 62 participantes no treino e 13 no teste. Acurácias T2/T3/T4: 69,23/61,54/69,23%. São resultados individuais, não fusão. O F1 reportado é da classe PD. |
| **Drotár et al., 2016** | Examina T1–T8 separadamente; combina T2–T8. Exclui T1 por dados de apenas 69 participantes e baixa discriminação observada. | Nota 3, p. 12; Tabelas 4–5, p. 14. SVM com cinemática e pressão: 81,3% de acurácia. Validação estratificada de dez folds repetida dez vezes. Há filtragem estatística de atributos; o texto não esclarece completamente seu aninhamento. [Texto dos autores](https://arxiv.org/pdf/2411.03044). |
| **Diaz et al., 2019** | Seleciona cinco tarefas pelo desempenho. Estática: T1/T3/T5/T7/T8; dinâmica: T2/T3/T6/T7/T8; imagem enriquecida: T1/T4/T5/T7/T8. Não há um conjunto único entre modalidades. | Seções 3.3–3.5, pp. 5–6; Tabelas 1–3, pp. 6–7, conferidas visualmente; lista da estática também em p. 8. Dez folds estratificados. Seleção de atributos explicitamente restrita ao treino; não fica estabelecida a mesma restrição para selecionar tarefas. Ensemble enriquecido: 86,67% de acurácia, Tabela 4, p. 9. [Texto dos autores](https://arxiv.org/pdf/2405.13438). |
| **Impedovo e Pirlo, 2019** | Revisão de análise dinâmica, incluindo a escolha das tarefas na aquisição. | É uma revisão metodológica, não um benchmark original que fixe um subconjunto PaHaW. Não usar sua citação para atribuir uma seleção experimental de tarefas. [Registro institucional e resumo dos autores](https://ricerca.uniba.it/handle/11586/218770), [IEEE](https://ieeexplore.ieee.org/document/8365754). |

Não adotar uma lista da literatura porque ela produz o maior número na nossa coorte. Em particular, reproduzir a seleção das cinco melhores tarefas requer fazê-la novamente **dentro do treino**; copiar seus resultados de validação como garantia de ganho seria inadequado.

## Rastreabilidade das fontes e limites de comparação

O TFG fornecido pelo usuário foi conferido em `D:/doutorado/PseudodynamicHandwritingMinimapSignatures/TFG_AlbertoCasademunt (1).pdf`. Seu [repositório original](https://github.com/vibrantsalt/TFG) permanece no commit `c721578e7e5acd8a9e0eda11aeca62a79902a5f8`. No [notebook desse commit](https://github.com/vibrantsalt/TFG/blob/c721578e7e5acd8a9e0eda11aeca62a79902a5f8/PD_Prediction.ipynb), a configuração declara compatibilidade com T2/T3/T4; a separação de participantes precede a produção dos recortes. O TFG consulta o grupo denominado teste durante a exploração dos tamanhos de recorte. Não há evidência observada de mistura de recortes do mesmo participante entre treino e teste nessa configuração.

Os três artigos já constam de `paper/refs.bib`. DOIs reconfirmados nos registros Crossref e nas fontes acima: [Drotár](https://doi.org/10.1016/j.artmed.2016.01.004), [Diaz](https://doi.org/10.1016/j.patrec.2019.08.018), [Impedovo e Pirlo](https://doi.org/10.1109/RBME.2018.2840679). Nenhuma referência bibliográfica nova é necessária. O TFG não tem DOI verificado; permanece como a referência fornecida expressamente pelo usuário.

A [fonte oficial PaHaW](https://bdalab.utko.fekt.vut.cz/) descreve 75 participantes, sinais durante contato e no ar e aquisição nominal de 150 Hz. As publicações divergem quanto à frequência nominal. Para a baseline cinemática, usar os timestamps presentes nos arquivos, com unidades documentadas, e não substituir os intervalos por uma frequência copiada de outro estudo. Um descritor inspirado na literatura, mas com definições ou processamento diferentes, deve ser chamado de baseline implementada neste estudo, não reprodução exata de Drotár.

No artigo, separar acurácia, macro F1, F1 da classe PD e fusão por participante. Comparações com números publicados continuam contextuais: protocolos, seleção, modalidades, recortes e conjuntos de tarefas diferem. Uma CNN com pesos pré-treinados também deve ser identificada como transferência de aprendizado; o resultado da CNN anterior treinada do zero não representa essa nova condição.
