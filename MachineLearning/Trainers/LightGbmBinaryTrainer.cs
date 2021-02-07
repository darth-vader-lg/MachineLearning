using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryModelParameters;
using TTransformer = Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.LightGbm.LightGbmBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;
using TTrainer = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer;
using TOptions = Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LightGbmBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LightGbmBinaryTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
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
