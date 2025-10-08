"""
Chat templates for different model types.
"""

FIXED_QWEN_TEMPLATE = r"""
{# ----- header / system ----- #}
{%- if messages and messages[0]['role'] == 'system' -%}
  {{- "<|im_start|>system\n" ~ messages[0]['content'] ~ "<|im_end|>\n" -}}
  {%- set system_offset = 1 -%}
{%- else -%}
  {{- "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" -}}
  {%- set system_offset = 0 -%}
{%- endif -%}

{# ----- main loop ----- #}
{%- for message in messages -%}
  {%- if message.role == 'system' and loop.first -%}
    {# Skip system message as it's already handled above #}
  {%- else -%}
    {# Check alternating pattern: account for system message offset #}
    {%- set adjusted_index = loop.index0 - system_offset -%}
    {%- if (message['role'] == 'user') != (adjusted_index % 2 == 0) -%}
      {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}
    
    {%- if message.role == 'user' or (message.role == 'system' and not loop.first) -%}
      {{- "<|im_start|>" ~ message.role ~ "\n" ~ message.content ~ "<|im_end|>\n" -}}
    {%- elif message.role == 'assistant' -%}
      {{- "<|im_start|>assistant\n" -}}
      {% generation %}{{- message.content -}}{% endgeneration %}
      {{- "<|im_end|>" -}}
      {{- "\n" if not loop.last else "" -}}
    {%- endif -%}
  {%- endif -%}
{%- endfor -%}

{# ----- generation prompt ----- #}
{%- if add_generation_prompt -%}
  {{- "<|im_start|>assistant\n" -}}
{%- endif -%}
"""

FIXED_GEMMA_TEMPLATE = r"""{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else "") }}
    {%- if role == 'model' -%}
        {% generation %}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
        {% endgeneration %}
    {%- else -%}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
    {%- endif -%}
    {{ '<end_of_turn>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{ '<start_of_turn>model\n' }}
{%- endif -%}
"""

FIXED_LLAMA_TEMPLATE = r"""
{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Main conversation loop #}
{%- for message in messages %}
    {%- if message.role == "user" %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message.content | trim + '<|eot_id|>' }}
    {%- elif message.role == "assistant" %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
        {% generation %}{{- message.content -}}{% endgeneration %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""