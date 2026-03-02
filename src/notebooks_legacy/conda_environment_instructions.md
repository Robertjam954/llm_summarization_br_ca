# How to Create and Use a Conda Environment for Jupyter Notebooks

## 1. Creating a New Conda Environment (First Time)

1. Open a terminal (Anaconda Prompt or VS Code terminal).
2. Run the following command to create a new environment (replace `llm_summary_env` and `python=3.11` as needed):

   ```sh
   conda create -n llm_summary_env python=3.11
   ```

3. Activate the environment:

   ```sh
   conda activate llm_summary_env
   ```

4. Install Jupyter:

   ```sh
   conda install jupyter
   ```

## 2. Activating the Environment in the Future

Whenever you open a new terminal or restart VS Code:

1. Open a terminal.
2. Activate your environment:

   ```sh
   conda activate llm_summary_env
   ```

## 3. Starting Jupyter Notebook

1. Make sure your environment is activated (`llm_summary_env`).
2. Navigate to your project folder if needed:

   ```sh
   cd path/to/llm_summary
   ```

3. Start Jupyter Notebook:

   ```sh
   jupyter notebook
   ```

4. In the Jupyter interface, you can create new notebooks and select the `llm_summary_env` kernel.

---

**Tip:** If you want to use this environment as a kernel in Jupyter, you may need to run:

```sh
conda install ipykernel
python -m ipykernel install --user --name llm_summary_env --display-name "Python (llm_summary_env)"
```

This will make your conda environment available as a selectable kernel in Jupyter.