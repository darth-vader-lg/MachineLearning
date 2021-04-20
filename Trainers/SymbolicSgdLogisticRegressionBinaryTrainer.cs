using Microsoft.ML;
using System;
using TOptions = Microsoft.ML.Trainers.SymbolicSgdLogisticRegressionBinaryTrainer.Options;
using TTrainer = Microsoft.ML.Trainers.SymbolicSgdLogisticRegressionBinaryTrainer;
using TTransformer = Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.LinearBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SymbolicSgdLogisticRegressionBinaryTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class SymbolicSgdLogisticRegressionBinaryTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal SymbolicSgdLogisticRegressionBinaryTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(Options);
      #endregion
   }
}
