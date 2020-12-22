namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i providers di opzioni di testo
   /// </summary>
   public interface ITextOptionsProvider
   {
      #region Properties
      /// <summary>
      /// Opzioni di caricamento dati testuali
      /// </summary>
      TextLoaderOptions TextOptions { get; }
      #endregion
   }
}
