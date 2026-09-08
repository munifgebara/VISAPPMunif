# Experimentos autorizados: seleção no treinamento e controles espaciais

O protocolo está em `DECISION.md`; os parâmetros, sementes, fontes e famílias de
comparações estão em `config.json`. Esses dois arquivos foram congelados antes
do ajuste dos modelos. Resultados anteriores não são sobrescritos.

São avaliados LPQ e SVM, mantendo as nove representações originais. A escolha
entre estática e dinâmica, os parâmetros SVM por tarefa e a regra de fusão ficam
restritos ao treinamento. OOF intermediárias para escolher fusão repetem a
seleção completa em subdivisões próprias. Os controles espaciais recebem sua
própria seleção entre as oito representações dinâmicas e avaliam apenas tarefas.

## Executar neste ambiente

O ambiente de Python existente fornece as versões fixadas no projeto. Para usar
o código deste checkout sem modificar a instalação de outros projetos:

```powershell
$env:PYTHONPATH = 'C:/Users/munif/Documents/ChatGPT/PseudodynamicHandwritingMinimapSignatures/src'
$taskPython = 'C:/Users/munif/PycharmProjects/PseudodynamicHandwritingMinimapSignatures/.venv/Scripts/python.exe'
& $taskPython scripts/prepare_spatial_control_features.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
& $taskPython scripts/run_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json --phase all
& $taskPython scripts/analyze_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
& $taskPython scripts/verify_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
& $taskPython scripts/finalize_nested_selection_controls.py --config experiments/2026-09-restart/nested-selection-controls-v1/config.json
```

Para outra máquina, ajustar os caminhos em uma cópia de configuração e usar uma
pasta de resultados nova. São necessários os dados PaHaW e os descritores originais
das três etapas indicadas na configuração. O código de preparação confere a
identidade das imagens e LPQ originais antes de gerar os controles. Não há
normalização aprendida com a coorte na geração dos descritores.

`--phase authentic` e `--phase controls` permitem executar as duas partes
separadamente. A retomada dos ajustes reaproveita somente unidades externas
concluídas, identificadas por `result.json`, e recusa configuração, fontes ou
entradas diferentes das congeladas. Não editar arquivos de ajuste durante uma
execução. A análise só aceita a campanha completa de 175 unidades externas.

## Ler os resultados

- `../runs/nested-selection-controls-v1/reports/nested_selection_controls_report.md`:
  resultados, interpretação e limites.
- `../runs/nested-selection-controls-v1/metrics/global_summary.csv`:
  média das tarefas e endpoint distinto de fusão por participante.
- `../runs/nested-selection-controls-v1/metrics/task_global_comparisons.csv`:
  quatro contrastes principais, com intervalos pareados e Holm.
- `../runs/nested-selection-controls-v1/metrics/seed_summary.csv`:
  todas as sementes, sem seleção ou ensemble de suas predições.
- `../runs/nested-selection-controls-v1/manifests/verification.json`:
  auditoria de separação, seleção, modelos salvos e agregação.
- `../runs/spatial-controls-v1/manifests/verification.json`:
  auditoria da preparação dos controles.

Os modelos, matrizes de atributos e imagens intermediárias ficam disponíveis
localmente e são excluídos do Git. As predições, buscas internas comprimidas,
diagnósticos, fontes congeladas, figuras e relatórios documentam a execução.
Esta campanha não altera automaticamente o artigo: os novos resultados precisam
ser interpretados antes de decidir como reorganizar suas conclusões.
