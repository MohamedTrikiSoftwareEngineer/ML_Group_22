ML_Group_22
Repeat Purchase Prediction, NOVA IMS

HOW TO RUN THE FOUR NOTEBOOKS


WHAT YOU NEED

Anaconda or Miniconda, with conda available in Anaconda Prompt or your terminal.
The four course data files: train.csv, test.csv, clientes.csv and sample_submission.csv.
Internet access for the first environment creation only.

The archive contains the four notebooks, the environment file and the empty folders the
notebooks write into. It does not contain the course data, so copy the four CSV files in
before running anything.


1. UNPACK THE ARCHIVE

Extract ML_Group22_Notebook.zip into a folder of your choice. Keep the folder structure as
it is. You should see:

   01_notebooks/
   data/
   artifacts/
      features/
      figures/
      logs/
      metrics/
   deliverables/
   environment.yml
   README.txt

Please keep those folders even though they are empty. The notebooks write into them but
never create them, so a missing subfolder stops the run with a FileNotFoundError.


2. ADD THE DATA

Copy train.csv, test.csv, clientes.csv and sample_submission.csv into data/.


3. CREATE THE ENVIRONMENT

Open Anaconda Prompt or a terminal in the folder that contains environment.yml and run:

   conda env create -f environment.yml

This creates an environment called ml_group_22 with the exact package versions used for the
results. It takes a few minutes the first time.


4. ACTIVATE THE ENVIRONMENT AND REGISTER ITS KERNEL

Run these two commands in the same folder:

   conda activate ml_group_22
   python -m ipykernel install --sys-prefix --name ml_group_22 --display-name "Python (ml_group_22)"

The second command makes the environment visible to Jupyter. Please run it once.


5. START JUPYTER FROM THE SAME FOLDER

Run:

   jupyter lab

Open the 01_notebooks folder. When a notebook opens, check the kernel name in the top right
corner and select "Python (ml_group_22)". The default "Python 3" kernel can belong to a
different installation, so please check the name before running anything.


6. RUN THE NOTEBOOKS IN THIS ORDER

   1. Homework_Group22_01_Exploration.ipynb
   2. Homework_Group22_02_Cleaning-Preprocessing.ipynb
   3. Homework_Group22_03_Feature-Selection.ipynb
   4. Homework_Group22_04_Modelling-Assessment.ipynb

Each notebook reads the files written by the one before it. Run all cells of a notebook and
wait for it to finish before opening the next one. The whole sequence takes about two
minutes on a laptop.


WHERE THE RESULTS GO

The intermediate CSV files go back into data/.
The figures, metrics, fitted preprocessing parameters and logs go into artifacts/.
The Kaggle submission is written to deliverables/ML_Group_22_Version05.csv.


NOTES

The notebooks were executed before submission, so every table and figure can also be read
without running anything.

The environment file pins exact versions. Please do not upgrade packages inside it, because
newer versions change the predicted probabilities and therefore the submitted file. The
earlier Python 3.11 set is kept as a comment at the top of environment.yml. It does not
reproduce the numbers we report, which is why the file now pins Python 3.14.

The repository also carries requirements.txt with the same pins, for anyone who prefers a
plain virtual environment over conda.

The working directory matters. Start Jupyter from the folder that contains 01_notebooks, so
each notebook finds the project root correctly.

If "Python (ml_group_22)" does not appear in the kernel list, activate the environment and
run the ipykernel command from step 4 again, then reload the Jupyter page.
