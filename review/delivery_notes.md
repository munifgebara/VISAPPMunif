# Entrega da reescrita integral

O artigo foi refeito em inglês no template SCITEPRESS, partindo da geometria estática
e avançando para sinais isolados, combinações RGB, LPQ com SVM, outros classificadores,
fusão de tarefas e XAI da CNN. O experimento de largura do traço foi excluído do texto
e da galeria. A espessura constante aparece apenas como parâmetro de renderização.

## Arquivos finais

| Arquivo | Conteúdo |
|---|---|
| `output/pdf/manuscript.pdf` | 12 páginas; 7 figuras principais, 3 tabelas e 12 referências citadas |
| `output/pdf/manuscript_with_figure_alternatives.pdf` | O mesmo artigo, seguido de 12 figuras alternativas nas páginas 13–24 |
| `output/pdf/manuscript_overleaf.zip` | Fontes LaTeX, bibliografia, template, figuras PDF/PNG e registros de revisão |

O resumo tem 171 palavras. A extração do PDF principal contabiliza 36.933 caracteres
sem espaços, incluindo texto extraível de tabelas, gráficos e referências. O log final
não contém referências indefinidas, caracteres ausentes ou caixas transbordando.
As 24 páginas foram renderizadas e abertas individualmente; os relatórios finais
de QA estão nesta pasta. O manifesto de build registra os hashes dos arquivos.

## Revisão realizada e incorporada

A primeira versão foi preservada em `review/first_draft/` no repositório dos
experimentos. A revisão científica conferiu os números nos resultados salvos; a
revisão editorial examinou estrutura, precisão das referências e apresentação.
Foram removidas duas tabelas que repetiam gráficos, corrigidas atribuições
bibliográficas, explicitados os limites da seleção da codificação, detalhados os
alvos de Grad-CAM e melhorados os rótulos das figuras. O conteúdo foi reduzido de
15 para 12 páginas sem diminuir fontes ou margens do template.

Os critérios de qualidade vieram de três artigos VISAPP, identificados com DOI em
`bibliography_audit.md`: definição explícita do método, figuras que respondem ao
argumento adjacente, distinção entre desempenho preditivo e qualidade de explicação,
e clareza sobre métricas e agregação. `revision_response.md` registra as alterações
em resposta aos pareceres; os dois relatórios finais de QA encerram suas pendências
de paginação, referências e legibilidade.

A bibliografia anterior foi mantida no arquivo `.bib`, usando no texto as referências
pertinentes. As três novas referências metodológicas têm DOI confirmado. O TFG de
Alberto Casademunt González, orientado por Ángel Sánchez Calle e solicitado pelo
usuário, foi identificado e comparado separadamente; não foi encontrado DOI para ele.
As comparações com o TFG, Drotár e Diaz distinguem métrica, tarefas e protocolo.

Não foi necessário treinar novos modelos para estas revisões. Os gráficos e o texto
usam os experimentos concluídos e os resultados de XAI já salvos. As conclusões
permanecem condicionais: SAZ não mostrou vantagem estatística clara sobre estática,
e o ranking após fusão não demonstrou diferença entre os classificadores.

Esta é a versão revisada para leitura dos autores e escolha das figuras. A galeria
permite trocar as ilustrações sem ampliar o corpo de 12 páginas. O registro sobre a
declaração de assistência por IA está em `ai_disclosure.md`; a submissão não foi
realizada nem aprovada em nome dos autores.
