namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i gestori di storage dei modelli
   /// </summary>
   public interface IModelStorage
   {
      #region Methods
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il modello</returns>
      CompositeModel LoadModel(IMachineLearningContextProvider context);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      void SaveModel(IMachineLearningContextProvider context, CompositeModel model);
      #endregion
   }
}