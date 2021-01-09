using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;

namespace MachineLearning.Util
{
   /// <summary>
   /// Comparatore di vettori per valori
   /// </summary>
   internal static class VectorsComparer
   {
      #region Fields
      /// <summary>
      /// Dizionario dei metodi
      /// </summary>
      private static Dictionary<string, MethodInfo> _comparer =
         (from m in typeof(VectorsComparer).GetMethods()
          where m.Name == nameof(CompareByValues) && m.IsGenericMethod
          select m).ToDictionary(m => m.GetParameters()[0].ParameterType.Name);
      #endregion
      #region Methods
      /// <summary>
      /// Compara due oggetti per valore
      /// </summary>
      /// <param name="obj1">Primo oggetto</param>
      /// <param name="obj2">Secondo oggetto</param>
      /// <returns></returns>
      public static bool CompareByValues(object obj1, object obj2)
      {
         // Verifica se deve comparare un VBuffer<T>
         var type1 = obj1.GetType();
         if (_comparer.TryGetValue(type1.IsArray ? "T[]" : type1.Name, out var method)) {
            if (method.MakeGenericMethod(type1.IsArray ? type1.GetElementType() : type1.GetGenericArguments()[0]) is var genericMethod && genericMethod != null)
               return (bool)genericMethod.Invoke(null, new[] { obj1, obj2 });
         }
         return obj1.Equals(obj2);
      }
      /// <summary>
      /// Compara due array per valore
      /// </summary>
      /// <typeparam name="T">Tipo di array</typeparam>
      /// <param name="obj">Oggetto da comparare</param>
      /// <param name="other">Oggetto di comparazione</param>
      /// <returns>true se oggetti uguali nel contenuto</returns>
      public static bool CompareByValues<T>(this T[] obj, T[] other)
      {
         if (obj == null || other == null)
            return obj == null && other == null;
         if (obj.Length != other.Length)
            return false;
         for (var i = 0; i < obj.Length; i++) {
            if (!CompareByValues(obj[i], other[i]))
               return false;
         }
         return true;
      }
      /// <summary>
      /// Compara due ReadOnlyMemory per valore
      /// </summary>
      /// <typeparam name="T">Tipo di ReadOnlyMemory</typeparam>
      /// <param name="obj">Oggetto da comparare</param>
      /// <param name="other">Oggetto di comparazione</param>
      /// <returns>true se oggetti uguali nel contenuto</returns>
      public static bool CompareByValues<T>(this ReadOnlyMemory<T> obj, ReadOnlyMemory<T> other) => CompareByValues(obj.ToArray(), other.ToArray());
      /// <summary>
      /// Compara due ReadOnlySpan per valore
      /// </summary>
      /// <typeparam name="T">Tipo di ReadOnlySpan</typeparam>
      /// <param name="obj">Oggetto da comparare</param>
      /// <param name="other">Oggetto di comparazione</param>
      /// <returns>true se oggetti uguali nel contenuto</returns>
      public static bool CompareByValues<T>(this ReadOnlySpan<T> obj, ReadOnlySpan<T> other) => CompareByValues(obj.ToArray(), other.ToArray());
      /// <summary>
      /// Compara due VBuffer per valore
      /// </summary>
      /// <typeparam name="T">Tipo di VBuffer</typeparam>
      /// <param name="obj">Oggetto da comparare</param>
      /// <param name="other">Oggetto di comparazione</param>
      /// <returns>true se oggetti uguali nel contenuto</returns>
      public static bool CompareByValues<T>(this VBuffer<T> obj, VBuffer<T> other) => CompareByValues(obj.GetValues().ToArray(), other.GetValues().ToArray());
      #endregion
   }
}
