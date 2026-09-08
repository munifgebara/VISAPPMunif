# Revisão após as três melhorias experimentais

Concluída em 8 de setembro de 2026 para a versão ampliada destinada aos
orientadores. A publicação foi autorizada pelo usuário na branch main.

## Experimentos integrados

O texto incorpora a seleção de codificação e fusão dentro do treinamento e os
controles espaciais executados na etapa anterior. Esta rodada acrescentou a
comparação competitiva em T2/T3/T4, fixadas pela referência de Casademunt antes
dos novos ajustes, com oito tarefas como avaliação secundária. A revisão
bibliográfica verificou que Drotár e Diaz usam conjuntos diferentes, sem atribuir
a qualquer referência uma recomendação universal de reduzir tarefas.

Foram comparados estática LPQ+SVM, seleção interna entre nove encodings com
LPQ+SVM, 83 características cinemáticas/pressão com SVM e ResNet18 ImageNet
congelada com cabeça logística L2. A definição de transferência foi explicitada:
não houve fine-tuning da rede convolucional em PaHaW.

Os novos números não foram usados para escolher outro experimento. A fusão
LPQ selecionada em T2/T3/T4 chega a 0,6449, contra 0,6143 da estática, com
incerteza pareada e ausência de suporte após Holm. A redução de tarefas ajuda
dois métodos numericamente e reduz dois. Todas as comparações declaradas,
inclusive desfavoráveis, aparecem nos CSVs e no relatório do experimento.

## Parecer aplicado

O parecer independente em competitive_manuscript_review.md conferiu números,
intervalos, acurácias individuais e a distinção entre endpoints. Foram aplicadas
as cinco correções recomendadas:

1. Identificação explícita do retorno à comparação inicial com SAZ fixo, também
   na seção/caption de fusão e no diagrama de partições.
2. Correção da interpretação do contraste 0,5991 versus 0,6171: são políticas de
   codificação sob seleção interna de fusão, e não um teste isolado da seleção
   da regra de fusão.
3. Distinção entre tempos derivados dos registros e redução de tempo total ou
   carga clínica, que não foi avaliada.
4. Restrição da afirmação sobre SVM sem sinais tabulares à comparação com LPQ.
5. Redução de catálogos numéricos repetidos na introdução e explicação direta
   de macro F1 versus AUC de frações de votos.

Foram ainda explicitados os pesos IMAGENET1K_V1, a votação sobre rótulos
preditos, a acurácia da fusão na tabela competitiva e a separação do XAI da
CNN anterior. Citações narrativas redundantes foram reduzidas. Os parágrafos
de ligação entre seções e subseções foram preservados.

## Rastreabilidade e apresentação

As duas etapas novas acompanham as oito anteriores no pacote de reprodução.
O pacote contém código, configurações e dez CSVs de resultados agregados;
não redistribui registros originais, predições individuais ou checkpoints.
ResNet18 já constava da bibliografia preservada, com DOI 10.1109/CVPR.2016.90.
Não foi acrescentada referência sem DOI a partir de uma busca nova; o TFG
continua como fonte expressamente fornecida pelo usuário.

O resumo possui 185 palavras. O artigo mantém o título escolhido e o diagrama
aprovado, agora identificado como a etapa inicial. Foram acrescentadas quatro
figuras principais e uma alternativa, totalizando 12 principais e 13 alternativas.
O rascunho tem 19 páginas e a seleção 32, com alternativas nas páginas 20–32.
A contagem extraível de 56.502 caracteres sem espaços excede o limite de
submissão, assim como a extensão excede a publicação padrão. Ambos permanecem
registrados para os cortes posteriores autorizados aos orientadores; o arquivo
é apresentado como versão ampliada, não como submissão pronta.

O build não apresenta referências indefinidas, caracteres ausentes ou caixas
transbordando. A inspeção individual abrangeu todas as 32 páginas, distribuída
entre os relatórios de QA e a conferência final do responsável pelo build.
Os relatórios históricos permanecem vinculados às respectivas versões.

As alterações de escrita tratam repetição, precisão e conexão entre argumentos;
não se usou detector de autoria. A assistência permanece declarada no registro
editorial ai_disclosure.md. Os resultados continuam insuficientes para afirmar
superioridade geral ou validação em outra coorte.
