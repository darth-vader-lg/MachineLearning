using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.FastTree.GamBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;
using TTrainer = Microsoft.ML.Trainers.FastTree.GamBinaryTrainer;
using TOptions = Microsoft.ML.Trainers.FastTree.GamBinaryTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe GamBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class GamBinaryTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal GamBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.Gam(Options);
      #endregion
   }
}
