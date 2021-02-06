using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.FastTree.FastTreeBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>;
using MSTrainer = Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastTreeRegressionTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastTreeBinaryTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastTreeBinaryTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.FastTree(Options);
      #endregion
   }
}
