using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.LinearSvmTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.LinearSvmTrainer;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.LinearSvmTrainer>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LbfgsLogisticRegressionBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LinearSvmTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LinearSvmTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.LinearSvm(Options);
      #endregion
   }
}
