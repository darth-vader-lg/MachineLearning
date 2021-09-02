using System;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Estensioni della SerializationInfo
   /// </summary>
   internal static class SerializationInfoExtensions
   {
      #region Methods
      /// <summary>
      /// Imposta ua proprieta' leggendolo dalle informazioni di serializzazione
      /// </summary>
      /// <typeparam name="T">Tipo di dato</typeparam>
      /// <param name="info">Informazioni di serializazione</param>
      /// <param name="name">Nome del dato nelle info</param>
      /// <param name="setter">Valore da impostare</param>
      /// <returns>Il valore impostato</returns>
      public static T Set<T>(this SerializationInfo info, string name, Func<T> getter, Action<T> setter)
      {
         setter((T)info.GetValue(name, typeof(T)));
         return getter();
      }
      /// <summary>
      /// Imposta un valore leggendolo dalle informazioni di serializzazione
      /// </summary>
      /// <typeparam name="T">Tipo di dato</typeparam>
      /// <param name="info">Informazioni di serializazione</param>
      /// <param name="name">Nome del dato nelle info</param>
      /// <param name="value">Valore da impostare</param>
      /// <returns>Il valore impostato</returns>
      public static T Set<T>(this SerializationInfo info, string name, out T value) => value = (T)info.GetValue(name, typeof(T));
      #endregion
   }
}
