# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
import os

os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(
            # model="Qwen/Qwen2.5-VL-7B-Instruct",
            # model="Qwen/Qwen2.5-0.5B-Instruct",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            enforce_eager=True,
        )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    # llm.start_profile()
    
    print('==============================')

    outputs = llm.generate(prompts, sampling_params)
    
    # llm.stop_profile()
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
