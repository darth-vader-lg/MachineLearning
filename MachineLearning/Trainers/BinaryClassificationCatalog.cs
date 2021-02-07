using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di classificazione binaria
   /// </summary>
   public class BinaryClassificationCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal BinaryClassificationCatalog(IContextProvider<MLContext> context) : base(context) { /*Context.BinaryClassification.Trainers.LinearSvm*/ }
      /// <summary>
      /// Restituisce un trainer di tipo AveragedPerceptron
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public AveragedPerceptronTrainer AveragedPerceptron(Microsoft.ML.Trainers.AveragedPerceptronTrainer.Options options = default) =>
         new AveragedPerceptronTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo FastForest
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastForestBinaryTrainer FastForest(Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer.Options options = default) =>
         new FastForestBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo FastTree
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastTreeBinaryTrainer FastTree(Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer.Options options = default) =>
         new FastTreeBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo FieldAwareFactorizationMachine
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(Microsoft.ML.Trainers.FieldAwareFactorizationMachineTrainer.Options options = default) =>
         new FieldAwareFactorizationMachineTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo Gam
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public GamBinaryTrainer Gam(Microsoft.ML.Trainers.FastTree.GamBinaryTrainer.Options options = default) =>
         new GamBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LbfgsLogisticRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LbfgsLogisticRegressionBinaryTrainer LbfgsLogisticRegression(Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options options = default) =>
         new LbfgsLogisticRegressionBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LdSvm
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LdSvmTrainer LdSvm(Microsoft.ML.Trainers.LdSvmTrainer.Options options = default) =>
         new LdSvmTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbm
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LightGbmBinaryTrainer LightGbm(Microsoft.ML.Trainers.LightGbm.LightGbmBinaryTrainer.Options options = default) =>
         new LightGbmBinaryTrainer(this, options);
      #endregion
   }
}
