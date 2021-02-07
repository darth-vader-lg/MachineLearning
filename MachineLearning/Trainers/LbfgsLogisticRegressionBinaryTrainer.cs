using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Trainers.LinearBinaryModelParameters>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LbfgsLogisticRegressionBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LbfgsLogisticRegressionBinaryTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LbfgsLogisticRegressionBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.LbfgsLogisticRegression(Options);
      #endregion
   }
}
