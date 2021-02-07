using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.LdSvmModelParameters;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.LdSvmModelParameters>;
using TTrainer = Microsoft.ML.Trainers.LdSvmTrainer;
using TOptions = Microsoft.ML.Trainers.LdSvmTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe LbfgsLogisticRegressionBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class LdSvmTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal LdSvmTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.LdSvm(Options);
      #endregion
   }
}
