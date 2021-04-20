namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per gli oggetti contenenti un contesto di machine learning
   /// </summary>
   public interface IMachineLearningContext
   {
      #region Properties
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      MachineLearningContext ML { get; }
      #endregion
   }
}
