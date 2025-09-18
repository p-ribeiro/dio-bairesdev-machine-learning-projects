# Virtual Assistance System

Assistente de voz simples para automações locais (pesquisa no YouTube, Wikipedia, contar piadas, dizer a hora, etc.).

## Requisitos

- Python 3.12+ (recomendado)
- Dependências listadas em `requirements.txt`

Instalar dependências:

```bash

pip install -r requirements.txt

```

## Como executar

Para iniciar o assistente (exemplo em português do Brasil):

```bash

python run_speech_automation.py pt-BR

```

Observação: este projeto define o idioma de sistema como `pt-BR` ao passar o argumento acima.

## Estrutura do projeto

- `run_speech_to_automation.py` / `run_speech_automation.py` — ponto de entrada para rodar o assistente por linha de comando
- `speech_to_automation/` — lógica principal e ações disponíveis
- `text_to_speech/`, `utils/`, `locales/` — recursos de áudio, utilitários e arquivos de tradução
- `requirements.txt` — dependências Python

## Uso

1. Conecte microfone e alto‑falante.
2. Execute o comando indicado para iniciar em português.
3. Diga comandos suportados (ex.: buscar no YouTube, procurar na Wikipedia, contar uma piada, dizer a hora, sair).

## Observações

- Ajuste permissões e drivers de áudio do sistema se necessário.
- Traduções e mensagens estão em `locales/`.

## Licença

Uso pessoal/experimental. Nenhuma garantia.
