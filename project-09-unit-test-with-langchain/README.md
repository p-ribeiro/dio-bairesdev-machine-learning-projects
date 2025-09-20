# Gerador Automático de Testes Unitários com LangChain + Azure OpenAI

Resumo
-----
Este documento descreve, como estudo e base para implementação futura, um agente de IA que recebe código Python (um trecho ou um arquivo) e produz automaticamente um arquivo de testes em Python puro usando pytest. A solução usa LangChain como orquestrador e Azure OpenAI como provedor do modelo de linguagem.

Objetivo
--------
Criar um agente que, dado um input Python, retorne um único arquivo .py contendo:
- primeira linha: `import pytest`
- várias funções `def test_*` cobrindo casos de sucesso e de falha (cenários positivos e negativos)
- nenhum texto adicional além do código Python válido (o output deve ser pronto para salvar como arquivo e executar com `pytest`).

Visão geral da arquitetura
--------------------------
- Interface que recebe código-fonte Python (texto ou uploaded file).
- Componente de pré-processamento: extrai assinaturas, docstrings e comportamentos esperados.
- LLM (Azure OpenAI) via LangChain: gera o conteúdo do arquivo de testes com base em um prompt controlado.
- Pós-processamento automático: valida o output (parsing AST, checagens básicas) e aplica correções se necessário.
- Resultado: string com o conteúdo do arquivo de testes pronto para salvar.

Por que LangChain + Azure OpenAI
-------------------------------
- LangChain: fornece abstrações (PromptTemplate, LLMChain, Agents) para organizar prompts, ferramentas e fluxos de controle.
- Azure OpenAI: provê modelos hospedados com integração corporativa e controle de custo/segurança.

Fluxo de execução desejado
--------------------------
1. Entrada: código Python (string) ou caminho para arquivo.
2. Extrair informações úteis:
   - nomes de funções/classes públicas
   - assinaturas (parâmetros padrão, tipos se disponíveis)
   - docstrings e comentários que indiquem comportamento esperado
3. Construir prompt estruturado que contenha:
   - instruções estritas de formato (apenas código Python, `import pytest` na primeira linha, nomes `test_*`)
   - exemplos curtos de input -> expected tests
4. Invocar LLM via LangChain (AzureOpenAI) com temperatura baixa para previsibilidade.
5. Receber output e aplicar validações:
   - garantir que primeira linha seja `import pytest`
   - garantir que o arquivo seja parseável (`ast.parse`)
   - pelo menos um `def test_` para cada função pública encontrada
   - gerar falhas controladas (mocks, inputs inválidos) quando aplicável
6. Retornar o conteúdo do arquivo de testes.

Prompt design (exemplo resumido)
--------------------------------
- Fornecer instruções rígidas sobre formato de saída.
- Incluir exemplos pequenos.
- Pedir comentários inline como `#` apenas quando estritamente úteis, preferir código claro.

Exemplo de instrução chave (resumida):
- "Receba o código Python a seguir. Gere apenas um arquivo de testes Python válido. A primeira linha deve ser `import pytest`. Crie funções `def test_<nome>_success():` e `def test_<nome>_failure():` para cada função pública. Não adicione texto explicativo fora do código. Use asserts e fixtures quando necessário."

Exemplo de implementação (esqueleto) em Python usando LangChain + Azure
---------------------------------------------------------------------
```python
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import AzureChatOpenAI  # ou AzureOpenAI conforme versão
import ast
import os

PROMPT_TEMPLATE = """
Você é um gerador de testes unitários. Regras:
- Responda APENAS com um arquivo Python válido.
- A PRIMEIRA linha deve ser: import pytest
- Para cada função pública do código abaixo, crie ao menos:
  - def test_<nome>_success():  # caso esperado
  - def test_<nome>_failure():  # caso de erro/entrada inválida
- Use asserts, fixtures ou mocks se necessário.
- Não coloque explicações, apóstrofos não relacionados ao código, ou texto extra.

Código de entrada:
```
{source_code}
```

Gere o arquivo de testes agora:
"""

def generate_tests(source_code: str) -> str:
    llm = AzureChatOpenAI(
        deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        openai_api_base=os.environ.get("AZURE_OPENAI_BASE"),
        openai_api_key=os.environ.get("AZURE_OPENAI_KEY"),
        openai_api_version=os.environ.get("AZURE_OPENAI_VERSION", "2023-10-01"),
        temperature=0.0
    )
    prompt = PromptTemplate(input_variables=["source_code"], template=PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    raw = chain.run(source_code)
    # Pós-processamento: garantir import pytest e AST válido
    if not raw.startswith("import pytest"):
        raw = "import pytest\n\n" + raw
    try:
        ast.parse(raw)
    except SyntaxError:
        # Aplicar heurística simples ou re-prompt (não detalhado aqui)
        raise
    return raw
```

Validações recomendadas
-----------------------
- Parsing com ast.parse para garantir código Python.
- Checar presença de `import pytest`.
- Verificar que cada função pública tem testes gerados.
- Executar pytest em modo isolado (subprocesso com timeout) como etapa opcional de sanity check.

Cuidados e limitações
---------------------
- Modelos podem produzir código incorreto; sempre validar automaticamente.
- Cobertura gerada não substitui revisão humana.
- Evitar exposição de segredos ao enviar código para LLMs.
- Monitorar custo e latência do Azure OpenAI; usar temperature=0 para consistência.

Integração CI/CD
----------------
- Gerar testes automaticamente como artefato provisório.
- Incluir etapa de lint/ast/pytest para bloquear commits que gerem testes inválidos.
- Rastrear mudanças geradas por IA em PRs separados para revisão humana.

Próximos passos (implementação futura)
--------------------------------------
- Implementar re-prompting e correção automática quando AST inválido.
- Criar heurísticas para gerar testes parametrizados e fixtures reutilizáveis.
- Implementar ferramenta local (CLI) que chama o agente e salva arquivos de teste.
- Adicionar métricas (taxa de aceitação humana, cobertura gerada) para avaliar qualidade.

Referências rápidas (configuração Azure)
---------------------------------------
- Defina variáveis de ambiente: AZURE_OPENAI_KEY, AZURE_OPENAI_BASE, AZURE_DEPLOYMENT_NAME, AZURE_OPENAI_VERSION.
- Use a classe adequada da versão do LangChain (AzureChatOpenAI ou AzureOpenAI) conforme a API disponível.

Conclusão
---------
Este README define um plano prático e controlado para construir um agente que converta código Python em testes pytest usando LangChain e Azure OpenAI. A implementação deve priorizar prompts estritos, validação automática e revisão humana antes de rodada em produção.
