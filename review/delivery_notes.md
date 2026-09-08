# Entrega da reescrita integral

Atualização final de 8 de setembro de 2026: revisão científica e editorial segundo
os critérios do VISAPP, com correções incorporadas sem novos experimentos.
A conclusão valoriza os ganhos observados e preserva sua incerteza; método e XAI
receberam esclarecimentos sobre inferência, mapas nulos e alcance das explicações.
Foram reduzidas repetições e frases genéricas. Permanecem o título escolhido,
a formulação proposta, os seis parágrafos de ligação e o diagrama aprovado como
Figura 1, na página 2. O rascunho ampliado é destinado à discussão dos cortes
com os orientadores Yandre e Ángel, conforme autorização do usuário.

O artigo foi refeito em inglês no template SCITEPRESS, partindo da geometria estática
e avançando para sinais isolados, combinações RGB, LPQ com SVM, outros classificadores,
fusão de tarefas e XAI da CNN. O experimento de largura do traço foi excluído do texto
e da galeria. A espessura constante aparece apenas como parâmetro de renderização.

## Arquivos finais

| Arquivo | Conteúdo |
|---|---|
| `output/pdf/manuscript.pdf` | 14 páginas; 8 figuras principais, 3 tabelas e 13 referências citadas |
| `output/pdf/manuscript_with_figure_alternatives.pdf` | O mesmo artigo, seguido de 12 figuras alternativas nas páginas 15–26 |
| `output/pdf/manuscript_overleaf.zip` | Fontes LaTeX, bibliografia, template, figuras PDF/PNG, SVG do diagrama, geradores, código/configurações dos oito estágios e registros de revisão |

O resumo tem 184 palavras. A contagem atual de caracteres está no manifesto de build,
incluindo texto extraível de tabelas, gráficos e referências. O log final
não contém referências indefinidas, caracteres ausentes ou caixas transbordando.
O relatório `final_referee_review.md` registra o parecer final e as correções;
`final_referee_methods_qa.md` e `final_referee_style_qa.md` registram o QA correspondente.
`section_bridges.md` documenta a inclusão anterior das transições.
`title_novelty_and_gains.md` e `opening_revision.md` documentam as etapas anteriores.
Os relatórios `activity_insertion_main_qa.md` e `activity_insertion_gallery_qa.md`
documentam a versão anterior à revisão da abertura. O manifesto de build registra os hashes
dos arquivos. A galeria é renumerada automaticamente após o corpo completo.

## Revisão realizada e incorporada

A primeira versão foi preservada em `review/first_draft/` no repositório dos
experimentos. A revisão científica conferiu os números nos resultados salvos; a
revisão editorial examinou estrutura, precisão das referências e apresentação.
Foram removidas duas tabelas que repetiam gráficos, corrigidas atribuições
bibliográficas, explicitados os limites da seleção da codificação, detalhados os
alvos de Grad-CAM e melhorados os rótulos das figuras. O conteúdo foi reduzido de
15 para 12 páginas sem diminuir fontes ou margens do template na primeira entrega.
O diagrama aprovado levou o rascunho a 13 páginas; a revisão da abertura levou-o
a 14. Os oito painéis principais permanecem no artigo, sem cortes para retornar
ao limite anterior.

Os critérios de qualidade vieram de três artigos VISAPP, identificados com DOI em
`bibliography_audit.md`: definição explícita do método, figuras que respondem ao
argumento adjacente, distinção entre desempenho preditivo e qualidade de explicação,
e clareza sobre métricas e agregação. `revision_response.md` registra as alterações
em resposta aos pareceres; os dois relatórios finais de QA encerram suas pendências
de paginação, referências e legibilidade.

A bibliografia anterior foi mantida no arquivo `.bib`, usando no texto as referências
pertinentes. As três referências metodológicas acrescentadas na reescrita têm DOI
confirmado. Cilia et al. (2021), também com DOI confirmado, foi acrescentado para
delimitar a contribuição frente a um antecedente de dinâmica codificada em RGB. O TFG de
Alberto Casademunt González, orientado por Ángel Sánchez Calle e solicitado pelo
usuário, foi identificado e comparado separadamente; não foi encontrado DOI para ele.
As comparações com o TFG, Drotár e Diaz distinguem métrica, tarefas e protocolo.

Não foi necessário treinar novos modelos para estas revisões. Os gráficos e o texto
usam os experimentos concluídos e os resultados de XAI já salvos. SAZ melhora o
macro F1 observado em quatro tarefas, incluindo a espiral (0.5915 para 0.6716),
e a média das tarefas (0.5643 para 0.5752). As conclusões permanecem condicionais:
a vantagem global sobre estática continua incerta, e o ranking após fusão não
demonstrou diferença entre os classificadores.

O pacote `reproducibility/experimental/` contém 37 fontes originais, README e
manifesto SHA-256. `experimental_reproduction.md` descreve os dados necessários,
ajustes de caminhos e ordem dos 16 comandos de execução e verificação. Os modelos
não foram executados nesta auditoria; sintaxe, dependências e integridade foram
verificadas. Registros PaHaW, predições individuais e checkpoints não são
redistribuídos, e os geradores finais de figuras dependem da consolidação histórica
documentada no README.

Esta é a versão revisada para leitura dos autores e escolha das figuras. A galeria
permite comparar alternativas enquanto os orientadores discutem o rascunho ampliado.
O registro sobre a
declaração de assistência por IA está em `ai_disclosure.md`; a submissão não foi
realizada nem aprovada em nome dos autores.
