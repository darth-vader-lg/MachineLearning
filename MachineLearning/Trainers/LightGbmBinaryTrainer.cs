using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.LightGbm.LightGbmBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LightGbmBinaryTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LightGbmBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.LightGbm(Options);
      #endregion
   }
}
