# LLMs vs BERT vs BiLSTM

Este repositório contém scripts Python para avaliar diferentes variantes de prompts em classificadores de sentimento usando modelos de linguagem local (`ollama`) e na nuvem (`OpenAI`). A lógica de prompts foi refatorada para um módulo centralizado em `src/constants/prompts.py`.

## Estrutura do projeto

- `run-llama-local.py` - roda inferência local usando o modelo `llama3.1:8b` via `ollama`.
- `run-gpt-cloud.py` - roda inferência via OpenAI Chat completions.
- `src/constants/prompts.py` - componente único que contém os prompts e templates usados pelos scripts.
- `ollama_artifacts/` - saída gerada pelo script local, incluindo métricas CSV e gráficos.
- `venv/` - ambiente virtual Python (não versionado, apenas presente no repositório local).

## Requisitos

- Python 3.9+ recomendado
- `ollama` instalado e configurado para executar `llama3.1:8b` localmente
- Conta OpenAI e variável de ambiente `OPENAI_API_KEY` definida para executar `run-gpt-cloud.py`

Dependências Python usadas:

- openai
- ollama
- datasets
- pandas
- numpy
- scikit-learn
- tqdm
- matplotlib
- seaborn

## Configuração

1. Crie e ative um ambiente virtual:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Instale as dependências:

```powershell
pip install openai ollama datasets pandas numpy scikit-learn tqdm matplotlib seaborn
```

3. Defina a chave da OpenAI (para `run-gpt-cloud.py`):

```powershell
$env:OPENAI_API_KEY = "sua_chave_aqui"
```

## Uso

### Executar inferência local com Ollama

```powershell
python run-llama-local.py
```

Os artefatos serão salvos em `ollama_artifacts/`, incluindo `ollama_metrics_summary.csv`, curvas ROC e matrizes de confusão.

### Executar inferência na nuvem com OpenAI

```powershell
python run-gpt-cloud.py
```

## Observações

- `run-gpt-cloud.py` importa os templates de prompt a partir de `src/constants/prompts.py`.
- Se houver problemas de importação de pacote, execute os scripts a partir da raiz do projeto ou defina `PYTHONPATH` para incluir a raiz.

## Prompts

Todos os prompts estão centralizados em `src/constants/prompts.py`. Isso facilita a manutenção e a experimentação com variantes de Zero-Shot, Few-Shot e Chain-of-Thought.
