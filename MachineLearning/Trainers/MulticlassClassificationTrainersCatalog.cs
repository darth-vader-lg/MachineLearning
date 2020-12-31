using ML = Microsoft.ML.Trainers;
using MLVision = Microsoft.ML.Vision;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers multiclasse
   /// </summary>
   public class MulticlassClassificationTrainersCatalog
   {
      #region Fields
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      private readonly MachineLearningContext _ml;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal MulticlassClassificationTrainersCatalog(MachineLearningContext ml) => _ml = ml;
      /// <summary>
      /// Restituisce un trainer di tipo SdcaNonCalibratedMulticlass
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaNonCalibratedMulticlassTrainer SdcaNonCalibrated(ML.SdcaNonCalibratedMulticlassTrainer.Options options = default) => new SdcaNonCalibratedMulticlassTrainer(_ml, options);
      /// <summary>
      /// Restituisce un trainer di tipo ImageClassification
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public ImageClassificationTrainer ImageClassification(MLVision.ImageClassificationTrainer.Options options = default) => new ImageClassificationTrainer(_ml, options);
      #endregion
   }
}
