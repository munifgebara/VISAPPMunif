# Revisão independente do protocolo competitivo

Data: 2026-09-08. Escopo: `competitive-task-subset-v1`, opção 3 e subconjunto de tarefas. Este documento acompanha a implementação; não substitui a auditoria das predições finais.

## Protocolo e descritores

Foram lidos `config.json`, `DECISION.md`, `kinematic_features.py`, `transfer_features.py` e os utilitários de seleção/comparação reutilizáveis. Não foi encontrado impedimento metodológico no protocolo declarado.

- **Tarefas e endpoint:** T2/T3/T4 é fixado por correspondência ao TFG de Casademunt, com fusão por participante como endpoint principal. T1–T8 é secundário. Os contrastes entre conjuntos comparam pipelines completos, pois também podem mudar a codificação escolhida. Uma média de F1 sobre três tarefas não deve ser apresentada como aumento direto da média sobre oito.
- **Seleção:** cada método pode selecionar sua codificação e seus parâmetros dentro dos mesmos participantes de treino. O SVM estático e a baseline cinemática têm codificações fixas; os dois métodos de imagem adaptativos escolhem entre nove. A fusão de votos é fixa e não exige outro nível de seleção. A implementação deverá comprovar que apenas os resultados internos das tarefas do conjunto declarado entram na escolha global.
- **Cinemática:** o código define 83 atributos: 11 globais e seis estatísticas para cada uma de 12 distribuições. A escala temporal é timestamp/1.000; coordenadas e pressão permanecem nas unidades nativas. Derivadas vetoriais são calculadas entre tempos médios consecutivos dentro de segmentos de estado constante. Intervalos não positivos ou maiores que 60 segundos interrompem derivadas e não contam como duração. As transições válidas são divididas igualmente entre contato e ar, uma convenção explícita. Tempos longos válidos fornecem velocidades médias sobre o intervalo, não movimento instantâneo observado.
- **Limites da baseline:** diferenças finitas sem suavização são sensíveis à quantização. Intervalos inválidos não criam falsos pen lifts, e distribuições ausentes são zeradas com diagnóstico registrado. São escolhas transparentes de uma baseline própria; não constituem reprodução exata de Drotár nem garantia de descritor cinemático ótimo.
- **Transferência:** a ResNet-18 remove a camada final, mantém todos os parâmetros congelados e módulos em modo de avaliação, calcula 512 ativações e usa normalização ImageNet fixa. O código verifica integridade dos pesos, repetibilidade da primeira inferência e imutabilidade dos buffers. Isso permite extração prévia por imagem, sem estimar estatísticas da coorte. O StandardScaler da cabeça aprendida deverá permanecer dentro do treino.
- **Interpretação da comparação:** a cabeça L2 linear, o SVM linear/RBF, o orçamento de busca e as resoluções de entrada são diferentes. O experimento compara quatro pipelines declarados. Não isola o efeito da arquitetura CNN, não é fine-tuning de ponta a ponta e não permite afirmar que todos os métodos de transferência foram explorados.
- **Inferência:** o protocolo exige reunir folds externos por repetição antes do cálculo de macro F1; depois, calcular a média das cinco repetições. O bootstrap e as trocas de predições devem conservar tarefas e repetições da mesma pessoa juntas. As famílias Holm foram declaradas e os intervalos são pontuais, condicionais às predições. A coorte já informava o desenvolvimento.

## Atenção comunicada à implementação

A configuração nova declara sementes e números de reamostragens distintos para bootstrap e randomização. O utilitário antigo `paired_task_mean_macro_f1_comparison` recebe um único `seed` e `resamples` e usa o mesmo gerador sequencialmente para ambos. Reutilizá-lo sem adaptação ou uma chamada apropriada violaria a configuração nova. Esse ponto foi comunicado antes da implementação da análise; ainda não representa uma falha observada em resultados.

## Seleção implementada

`competitive_selection.py` foi lido após a revisão inicial. `tune_family` valida que os três folds internos particionam exatamente os IDs externos de treino, e usa apenas seus índices no ajuste e no escore. `select_for_tasks` calcula a média apenas das tarefas solicitadas, preservando a ordem declarada no desempate. A escolha é gravada em `selection_before_test.json` antes da inferência externa. O cache de buscas e modelos utiliza família, aprendiz, codificação, tarefa e candidato; não reaproveita indevidamente uma escolha global entre conjuntos. Os modelos incluem StandardScaler ajustado no treino. A votação agrupa pessoa, conjunto, método e divisão, com empate favorável à classe PD conforme declarado.

A regressão logística usa `l1_ratio=0.0`, que corresponde à regularização L2 na versão instalada do scikit-learn (1.9.0). Não foi observado defeito no código de seleção lido. Isso ainda não comprova a integridade dos arquivos de entrada ou dos resultados finais.

## Auditoria da execução de ajuste

O runner final foi lido. `verify_competitive_task_subset.py --phase fits` terminou sem falhas: 237.368 verificações, 25 unidades completas, 16.440 predições por tarefa e 3.000 predições de fusão. Foram reconstruídos os escores a partir dos três valores de fold de cada candidato, todos os vencedores e desempates e todas as fusões. A auditoria confrontou as partições originais, os IDs de treino/validação/teste, os metadados, as grades completas, os manifestos de extração, os sinais de origem da cinemática e os hashes congelados.

Nas unidades determinísticas `r01_f01` e `r05_f05`, as predições e margens foram reproduzidas a partir dos 73 modelos persistidos, sem novo ajuste. As médias, variâncias e contagens dos StandardScaler correspondem aos dados de treino. A primeira passagem do auditor calculava a média LPQ em float32; esse cálculo independente foi corrigido para acumulação float64, como no StandardScaler, antes da passagem final. Não foi necessário corrigir os modelos ou repetir experimentos.

Registro: `experiments/2026-09-restart/runs/competitive-task-subset-v1/manifests/verification_fits.json`.

## Auditoria completa da análise

A análise final usa funções próprias que respeitam as sementes e os números de reamostragens separados. A observação preventiva sobre o utilitário antigo foi atendida. Os folds são agrupados por tarefa/repetição antes do cálculo de métricas. A hipótese de troca de métodos é aplicada por participante, conservando as cinco repetições e todas as tarefas juntas.

`verify_competitive_task_subset.py --phase complete` terminou sem falhas em 2026-09-08, com 237.893 verificações. Foram reconstruídos os 16 resumos globais, os escores por repetição e tarefa, os 28 contrastes declarados e as cinco famílias Holm (quatro de seis contrastes e uma de quatro). Os seis contrastes primários de fusão T2/T3/T4 tiveram seus intervalos de bootstrap e p-valores reproduzidos por uma implementação independente baseada em índices reamostrados e matrizes de troca, com as sementes fixadas. Nenhum modelo foi reajustado.

Limite: os intervalos e p-valores brutos dos contrastes secundários não foram regenerados independentemente; seus efeitos observados, cobertura de comparações e correções Holm foram verificados. Registro completo: `experiments/2026-09-restart/runs/competitive-task-subset-v1/manifests/verification.json`, com `complete_run_verified=true`.
