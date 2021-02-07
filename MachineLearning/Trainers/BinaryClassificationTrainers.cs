using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di classificazione binaria
   /// </summary>
   public class BinaryClassificationTrainers : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal BinaryClassificationTrainers(IContextProvider<MLContext> context) : base(context) { }
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
      /// <summary>
      /// Restituisce un trainer di tipo LightGbm
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public LinearSvmTrainer LinearSvm(Microsoft.ML.Trainers.LinearSvmTrainer.Options options = default) =>
         new LinearSvmTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo LightGbm
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public PriorTrainer Prior((string LabelColumnName, string FeaturesColumnName) options = default) =>
         new PriorTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo SdcaLogisticRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaLogisticRegressionBinaryTrainer SdcaLogisticRegression(Microsoft.ML.Trainers.SdcaLogisticRegressionBinaryTrainer.Options options = default) =>
         new SdcaLogisticRegressionBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo SdcaLogisticRegression
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaNonCalibratedBinaryTrainer SdcaNonCalibrated(Microsoft.ML.Trainers.SdcaNonCalibratedBinaryTrainer.Options options = default) =>
         new SdcaNonCalibratedBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo SgdCalibratedTrainer
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SgdCalibratedTrainer SgdCalibrated(Microsoft.ML.Trainers.SgdCalibratedTrainer.Options options = default) =>
         new SgdCalibratedTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo SgdNonCalibratedTrainer
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SgdNonCalibratedTrainer SgdNonCalibrated(Microsoft.ML.Trainers.SgdNonCalibratedTrainer.Options options = default) =>
         new SgdNonCalibratedTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo SymbolicSgdLogisticRegressionBinary
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SymbolicSgdLogisticRegressionBinaryTrainer SymbolicSgdLogisticRegression(Microsoft.ML.Trainers.SymbolicSgdLogisticRegressionBinaryTrainer.Options options = default) =>
         new SymbolicSgdLogisticRegressionBinaryTrainer(this, options);
      #endregion
   }
}
