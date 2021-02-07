using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.FastTree.FastForestBinaryModelParameters;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.FastTree.FastForestBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;
using TTrainer = Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer;
using TOptions = Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe FastTreeBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class FastTreeBinaryTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal FastTreeBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.FastTree(Options);
      #endregion
   }
}
